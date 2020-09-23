from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler

import torch
from torch.nn import Conv2d, Module, Linear, BatchNorm1d, Softmax, Dropout2d
import torch.nn.functional as F

from src.models.mask_r_cnn import cfg

backbone = build_backbone(cfg)

pooler = ROIPooler(output_size=(14, 14),
                   scales=[1.0],
                   sampling_ratio=1,
                   pooler_type='ROIPool',
                   canonical_box_size=224,
                   canonical_level=4)
# rois = pooler(backbone(NCHW_images), [kx4 tensors])


# TODO:
#  Just re-use the RoI Pooled features from mask head because after the second stage, features from RoI Align already
#  contain mask information.
#  -- Conv1: 3x3 conv - 256 channels -> ReLU
#  -- Conv2: 3x3 conv - 256 channels -> ReLU
#  -- Conv3: 3x3 conv - 256 channels -> ReLU
#  -- Conv4: 3x3 conv - 1024 channels -> ReLU
#  -- Pooling: GlobalAvgPool
#  -- ReLU
#  -- FC: 1024 to 256 channels (No ReLU) +BN
#  Besides, the similarity learning net have:
#  -- Subtraction (output 256 channels)
#  -- Element-wise square (output 256 channels)
#  -- FC: 256 to 2 channels (No ReLU)
#  (The first channel means similarity, the second channel means difference. Positive pair label (1,0) ,negative pair
#  label(0,1)
#  -- Softmax function.


class FeatureExtractor(Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv1_dropout = Dropout2d(p=0.4)
        self.conv2 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_dropout = Dropout2d(p=0.4)
        self.conv3 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_dropout = Dropout2d(p=0.4)
        self.conv4 = Conv2d(256, 1024, kernel_size=3, padding=1)
        self.conv4_dropout = Dropout2d(p=0.4)
        self.fc = Linear(1024, 256)
        self.batch_norm = BatchNorm1d(256)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv1_dropout(out)
        out = torch.relu(self.conv2(out))
        out = self.conv2_dropout(out)
        out = torch.relu(self.conv3(out))
        out = self.conv3_dropout(out)
        out = torch.relu(self.conv4(out))
        out = self.conv4_dropout(out)
        out = torch.relu(F.adaptive_avg_pool2d(out, 1024))
        out = out.view(-1, 1024)
        out = self.fc(out)
        out = self.batch_norm(out)
        return out


class SimilarityNet(Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        self.fc = Linear(256, 2)
        self.act = Softmax()

    def forward(self, x1, x2):
        out = x1 - x2
        out = out**2
        out = self.fc(out)
        out = self.act(out)
        return out


class MatchingNet(Module):
    def __init__(self):
        super(MatchingNet, self).__init__()
        self.fe = FeatureExtractor()
        self.sn = SimilarityNet()

    def forward(self, x1, x2):
        out1 = self.fe(x1)
        out2 = self.fe(x2)
        out = self.sn(out1, out2)
        return out
