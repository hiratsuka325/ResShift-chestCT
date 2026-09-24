import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cubical_complex import CubicalComplex
from models.PDMatching import WassersteinDistance
import numpy as np
from skimage.filters import frangi

def extract_filtered_pd(info, image, threshold=0.01):
    """
    PersistenceInformationからFrangi選別後の
    0次Persistence Diagramを返す
    """

    diagram = info.diagram
    pairing = info.pairing

    frangi_img = frangi(image)

    filtered_indices = []

    for i in range(len(diagram)):

        birth = diagram[i, 0]
        death = diagram[i, 1]

        birth_y = pairing[i, 0]
        birth_x = pairing[i, 1]

        persistence = abs(death - birth) / np.sqrt(2)

        weight = persistence * frangi_img[birth_y, birth_x]

        if weight > threshold:
            filtered_indices.append(i)

    if len(filtered_indices) == 0:
        return diagram[:0]

    return diagram[filtered_indices]

class wassersteinLoss_gaku(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        # Cubical complex constructor for persistent homology computation
        self.getPersistentInfo = CubicalComplex(dim=2)
        
        # distance between persistent diagrams
        self.criterion = WassersteinDistance(p=p)

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

        pi_x = self.getPersistentInfo(input)
        if self.precal_PD:  # read ground truth persistent diagram from pre-computed
            pi_y = [self.PD_target[img_names[0]][0]]
            for idx in range(1, N):
                pi_y.append(self.PD_target[img_names[idx]][0])
        else:
            pi_y = self.getPersistentInfo(target)

        for i in range(N):
            # 0-th persistent diagram (connected components)
            pd_x_0 = extract_filtered_pd(pi_x[i][0][0], input[i, 0], threshold=0.01)
            pd_y_0 = extract_filtered_pd(pi_y[i][0][0], target[i, 0], threshold=0.01)

            wd_0 = self.criterion(pd_x_0, pd_y_0)
        
            loss += wd_0

        loss /= N

        return loss
