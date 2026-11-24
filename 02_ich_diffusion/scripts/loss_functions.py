import torch
import torch.nn as nn
import torch.nn.functional as F




class CenterWeightedMSE(nn.Module):
    """
    pred / target shape: [N, C, H, W, D]
    center_fraction: fraction of each spatial dim that is considered "center"
                     e.g. 0.5 → central half in H, W, and D
    center_factor:   how much more to weight the center region (>1 emphasizes center)
    """
    def __init__(self, center_factor=2.0, center_fraction=0.5):
        super().__init__()
        self.center_factor = float(center_factor)
        self.center_fraction = float(center_fraction)

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have same shape"
        # pred/target: [N, C, H, W, D]
        weight = self._make_weights(pred)          # same shape & device as pred
        weight = weight.to(pred.device) 
        sq_err = (pred - target) ** 2              # [N, C, H, W, D]
        weighted_sq_err = sq_err * weight
        return weighted_sq_err.mean()

    def _make_weights(self, x):
        # x: [N, C, H, W, D]
        _, _, H, W, D = x.shape

        # start with all ones, same device/dtype as x
        weight = torch.ones_like(x)

        def central_slice(size):
            csize = int(size * self.center_fraction)
            csize = max(csize, 1)  # avoid empty center
            start = (size - csize) // 2
            end = start + csize
            return start, end

        h0, h1 = central_slice(H)
        w0, w1 = central_slice(W)
        d0, d1 = central_slice(D)

        # boost the central cube for all batches & channels
        weight[:, :, h0:h1, w0:w1, d0:d1] *= self.center_factor
        return weight