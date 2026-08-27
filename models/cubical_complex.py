"""Cubical complex computation module, adapted from the torch_topological library"""
import numpy as np
import torch
from torch import nn

from torch_topological.nn import PersistenceInformation
import gudhi
from concurrent.futures import ProcessPoolExecutor

def _gudhi_persistence_worker(args):
    x_np, superlevel = args

    if superlevel:
        x_np = -x_np

    cubical_complex = gudhi.CubicalComplex(
        dimensions=x_np.shape,
        top_dimensional_cells=x_np.flatten()
    )

    cubical_complex.persistence()
    
    cofaces = cubical_complex.cofaces_of_persistence_pairs()

    return cofaces

class CubicalComplex(nn.Module):
    def __init__(self, superlevel=False, dim=2):
        super().__init__()
        self.superlevel = superlevel
        self.dim = dim

    def forward(self, x):
        # Dimension was provided; this makes calculating the *effective*
        # dimension of the tensor much easier: take everything but the
        # last `self.dim` dimensions.
        if self.dim is not None:
            shape = x.shape[:-self.dim]
            dims = len(shape)

        # No dimension was provided; just use the shape provided by the
        # client.
        else:
            dims = len(x.shape) - 2

        # No additional dimensions present: a single image
        if dims == 0:
            return self._forward(x)

        # Handle image with channels, such as a tensor of the form `(C, H, W)`
        elif dims == 1:
            return [
                self._forward(x_) for x_ in x
            ]

        # Handle image with channels and batch index, such as a tensor of
        # the form `(B, C, H, W)`.
        elif dims == 2:
            
            batch_size = x.shape[0]
            channel_size = x.shape[1]
            
            assert channel_size == 1, \
                "This implementation assumes C=1."
                
            inputs = []

            for i in range(batch_size):
                x_img = x[i, 0]
                x_np = x_img.detach().cpu().numpy()
                inputs.append((x_np, self.superlevel))
                
            with ProcessPoolExecutor(max_workers=8) as executor:
                cofaces_list = list(
                    executor.map(
                        _gudhi_persistence_worker,
                        inputs
                    )
                )
                
            output = []

            for i in range(batch_size):

                x_img = x[i, 0]

                # superlevelの場合はPDの値についても
                # GUDHIと同じ変換をする
                if self.superlevel:
                    x_for_pd = -x_img
                else:
                    x_for_pd = x_img

                cofaces = cofaces_list[i]

                image_persistence_information = []

                max_dim = len(x_img.shape)

                for dim in range(max_dim):

                    pi = self._extract_generators_and_diagrams(
                        x_for_pd,
                        cofaces,
                        dim
                    )

                    image_persistence_information.append(pi)

                # 元の形
                # [batch][channel][PersistenceInformation]
                output.append(
                    [image_persistence_information]
                )

            return output

        else:
            raise ValueError(
                f"Unsupported input dimensions: {x.shape}"
            )        

    def _forward(self, x):
            if self.superlevel:
                x = -x
    
            cubical_complex = gudhi.CubicalComplex(
                dimensions=x.shape,
                top_dimensional_cells=x.flatten()
            )
    
            # We need the persistence pairs first, even though we are *not*
            # using them directly here.
            cubical_complex.persistence()
            cofaces = cubical_complex.cofaces_of_persistence_pairs()
    
            max_dim = len(x.shape)
    
            persistence_information = [
                self._extract_generators_and_diagrams(
                    x,
                    cofaces,
                    dim
                ) for dim in range(0, max_dim)
            ]
    
            return persistence_information

    def _extract_generators_and_diagrams(self, x, cofaces, dim):
        pairs = torch.empty((0, 2), dtype=torch.long)

        try:
            regular_pairs = torch.as_tensor(
                cofaces[0][dim], dtype=torch.long
            )
            pairs = torch.cat(
                (pairs, regular_pairs)
            )
        except IndexError:
            pass

        try:
            infinite_pairs = torch.as_tensor(
                cofaces[1][dim], dtype=torch.long
            )
        except IndexError:
            infinite_pairs = None

        if infinite_pairs is not None:
            # 'Pair off' all the indices
            max_index = torch.argmax(x)
            fake_destroyers = torch.empty_like(infinite_pairs).fill_(max_index)

            infinite_pairs = torch.stack(
                (infinite_pairs, fake_destroyers), 1
            )

            pairs = torch.cat(
                (pairs, infinite_pairs)
            )

        return self._create_tensors_from_pairs(x, pairs, dim)

    # Internal utility function to handle the 'heavy lifting:'
    # creates tensors from sets of persistence pairs.
    def _create_tensors_from_pairs(self, x, pairs, dim):

        xs = x.shape

        # Notice that `creators` and `destroyers` refer to pixel
        # coordinates in the image.
        creators = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 0], xs)
                ),
                dtype=torch.long
        )
        destroyers = torch.as_tensor(
                np.column_stack(
                    np.unravel_index(pairs[:, 1], xs)
                ),
                dtype=torch.long
        )
        gens = torch.as_tensor(torch.hstack((creators, destroyers)))

        persistence_diagram = torch.stack((
            x.ravel()[pairs[:, 0]],
            x.ravel()[pairs[:, 1]]
        ), 1)

        return PersistenceInformation(
                pairing=gens,
                diagram=persistence_diagram,
                dimension=dim
        )
