from .l2norm import L2Norm
from .multibox_loss import MultiBoxLoss
from .focal_loss import FocalLoss
from .GIoU import GiouLoss


# __all__ = ['L2Norm', 'MultiBoxLoss'] L2是正则化
__all__ = ['L2Norm', 'FocalLoss']
