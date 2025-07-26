import torch.nn as nn
import config

class RPN(nn.Module):
    def __init__(self, in_channels= 512, feature_channels= 512, num_anchors_per_location= config.NUM_ANCHORS_PER_LOC):  # 5 scales × 3 ratios
        super(RPN, self).__init__()
        self.conv= nn.Conv2d(in_channels, feature_channels, 3, padding= 1)
        self.cls= nn.Conv2d(feature_channels, num_anchors_per_location, 1)
        self.reg= nn.Conv2d(feature_channels, 4 * num_anchors_per_location, 1)
        self.relu= nn.ReLU(inplace= True)
    def forward(self, x):
        x= self.relu(self.conv(x))
        cls_logits= self.cls(x)
        reg_deltas= self.reg(x)

        batch_size, _, height, width= cls_logits.shape
        cls_logits= cls_logits.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        reg_deltas= reg_deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        return cls_logits, reg_deltas