import torch
import torch.nn as nn

class SimplifiedFasterRCNN(torch.nn.Module):
    def __init__(self, backbone, transform, rpn):
        super(SimplifiedFasterRCNN, self).__init__()
        self.backbone = nn.Sequential(*backbone)
        self.transform = transform
        self.rpn = rpn

    def override_nr_rpn_to_return(self):
        return 100

    def forward(self, x_in):
        im_list,_ = self.transform(x_in)
        x = self.backbone(im_list.tensors)
        self.rpn.post_nms_top_n = self.override_nr_rpn_to_return
        x = self.rpn(im_list, x)
        print(len(x[0][0]))
        return x