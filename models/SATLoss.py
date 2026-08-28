import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cubical_complex import CubicalComplex
from models.PDMatching import SpatialAware_WassersteinDistance
import multiprocessing

def _pd_loss_worker(args):
    """
    1枚の画像について
    GUDHI → H0 persistent diagram → Wasserstein distance
    を計算するworker
    """

    input_np, target_np, H, W, p = args

    # NumPy → CPU Tensor
    input_img = torch.from_numpy(input_np).float()
    target_img = torch.from_numpy(target_np).float()

    # (H, W) → (1, 1, H, W)
    input_img = input_img.unsqueeze(0).unsqueeze(0)
    target_img = target_img.unsqueeze(0).unsqueeze(0)

    # CubicalComplex
    getPersistentInfo = CubicalComplex(dim=2)

    # Persistent homology
    pi_x = getPersistentInfo(input_img)
    pi_y = getPersistentInfo(target_img)

    # H0だけ使用
    pd_x_0 = pi_x[0][0]
    pd_y_0 = pi_y[0][0]

    # Wasserstein distance
    criterion = SpatialAware_WassersteinDistance(p=p)

    wd_0 = criterion(
        pd_x_0,
        pd_y_0,
        H,
        W
    )

    # multiprocessingでTensorを返す必要はない
    return wd_0.item()

class PDMatchingLoss(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        # Cubical complex constructor for persistent homology computation
        self.getPersistentInfo = CubicalComplex(dim=2)

        # distance between persistent diagrams
        self.criterion = SpatialAware_WassersteinDistance(p=p)

        # For precomputed ground truth persistent diagram.
        self.precal_PD = opt.precal_PD
        self.PD_target = {}
        self.pad_dims = (1, 1, 1, 1)

    def _pad_to_square(self, x1, x2, H, W):
        margin = abs(H - W)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H > W:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if x1 is not None:
            x1 = F.pad(x1, paddings, "constant", 0.0)
        if x2 is not None:
            x2 = F.pad(x2, paddings, "constant", 0.0)

        return x1, x2

    def _pre_compute_PD(self, target, img_names):
        """
            Pre-compute PD for ground truth and save the training time.
        """
        # pad the boundary of the images by 1
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, _, H, W = target.size()

        # pad the image to square
        if H != W:
            _, padded_target = self._pad_to_square(None, padded_target, H, W)

        padded_target = torch.clamp(padded_target, min=0.0, max=1.0)
        padded_target = 1.0 - padded_target

        for i in range(N):
            img = padded_target[i,0,:,:].unsqueeze(0).unsqueeze(0)
            self.PD_target[img_names[i]] = self.getPersistentInfo(img)

    def forward(self, input, target, img_names=None):
        N, C, H, W = input.size()
        assert input.size() == target.size()
        assert input.device == target.device
        assert C == 1

        self.device = input.device

        input = input.to(torch.float32)
        target = target.to(torch.float32)

        # pad the boundary of the images by 1 (see Hu et al. NIPS 19' for reasons)
        padded_input = F.pad(input, self.pad_dims, mode='constant', value=1)
        padded_target = F.pad(target, self.pad_dims, mode='constant', value=1)

        N, C, H, W = input.size()

        # pad the image to square
        if H != W:
            input, target = self._pad_to_square(padded_input, padded_target, H, W)

        input = torch.clamp(input, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)

        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        # invert the image color to fit the computation in CubicalComplex (super-level filtration)
        input = 1.0 - input
        target = 1.0 - target

        # --------------------------------------------------
        # 画像ごとにworkerへ渡すデータを作る
        # --------------------------------------------------

        inputs = []

        for i in range(N):
            # 1枚の画像
            input_np = (input[i, 0].detach().cpu().numpy())
            target_np = (target[i, 0].detach().cpu().numpy())

            inputs.append(
                (input_np, target_np, H, W, 2)
            )
            
        # --------------------------------------------------
        # 画像単位で並列計算
        # --------------------------------------------------

        num_workers = min(8, N)

        with multiprocessing.Pool(processes=num_workers) as pool:

            losses = pool.map(
                _pd_loss_worker,
                inputs
            )
            
        loss_value = sum(losses) / N
        
        loss = torch.tensor(
            loss_value,
            dtype=torch.float32,
            device=self.device
        )
        
        return loss    
