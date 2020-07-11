# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from ..box_utils import match_gious,bbox_overlaps_giou,decode

class GiouLoss(nn.Module):
    """
        This criterion is a implemenation of Giou Loss, which is proposed in 
        Generalized Intersection over Union Loss for: A Metric and A Loss for Bounding Box Regression.

            Loss(loc_p, loc_t) = 1-GIoU

        The losses are summed across observations for each minibatch.

        Args:
            size_sum(bool): By default, the losses are summed over observations for each minibatch.
                                However, if the field size_sum is set to False, the losses are
                                instead averaged for each minibatch.
            predmodel(Corner,Center): By default, the loc_p is the Corner shape like (x1,y1,x2,y2)
            The shape is [num_prior,4],and it's (x_1,y_1,x_2,y_2)
            loc_p: the predict of loc
            loc_t: the truth of boxes, it's (x_1,y_1,x_2,y_2)
            
    """
    def __init__(self,pred_mode = 'Center',size_sum=True,variances=None):
        super(GiouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
    def forward(self, loc_p, loc_t,prior_data):
        num = loc_p.shape[0] 
        
        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        #loss = torch.tensor([1.0])
        gious =1.0 - bbox_overlaps_giou(decoded_boxes,loc_t)
        
        loss = torch.sum(gious)
     
        if self.size_sum:
            loss = loss
        else:
            loss = loss/num
        return 5*loss

