import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch.nn.functional as F


class ViTb16CHR(nn.Module):
    def __init__(self, model, preprocessing, num_classes):
        super(ViTb16CHR, self).__init__()

        self.model = model
        self.preprocessing = preprocessing

        self.cov4 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
        self.cov3 = nn.Conv2d(3072, 1024, kernel_size=1, stride=1)
        self.cov2 = nn.Conv2d(1536, 512, kernel_size=1, stride=1)

        self.cov3_1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.cov2_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.po1 = nn.AdaptiveAvgPool2d(1)
        self.po2 = nn.AdaptiveAvgPool2d(1)
        self.po3 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return torch.cat([k,y],1)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.model.parameters()},
                {'params': self.fc1.parameters()},
                {'params': self.fc2.parameters()},
                {'params': self.fc3.parameters()},
                {'params': self.cov4.parameters()},
                {'params': self.cov3.parameters()},
                {'params': self.cov2.parameters()},
                {'params': self.cov3_1.parameters()},
                {'params': self.cov2_1.parameters()},
                {'params': self.po1.parameters()},
                {'params': self.po2.parameters()},
                {'params': self.po3.parameters()}]

    def forward(self, x):
        # Apply preprocessing to the input
        x = self.preprocessing(x)

        # Extract features from ViT
        x = self.model(x)

        # Ensure x has the correct shape for Conv2d
        if x.dim() == 3:  # If ViT output is 3D (tokens), need reshaping
            batch_size = x.size(0)
            num_tokens = x.size(1)
            token_dim = x.size(2)
            # Assuming tokens are 1D and reshaping to 2D for Conv2d
            x = x.view(batch_size, num_tokens, token_dim, 1)
            x = x.permute(0, 3, 2, 1)  # Change to (N, C, H, W) format

        l4 = self.cov4(x)
        l4 = F.relu(l4)
        l4_1 = self.po1(l4)
        l4_2 = l4_1.view(l4_1.size(0), -1)
        o1 = self.fc1(l4_2)

        l3 = self.cov3_1(x)
        l3 = F.relu(l3)
        l3_1 = self._upsample_add(l4, l3)
        l3_2 = self.cov3(l3_1)
        l3_2 = F.relu(l3_2)
        l3_3 = self.po2(l3_2)
        l3_4 = l3_3.view(l3_3.size(0), -1)
        o2 = self.fc2(l3_4)

        l2 = self.cov2_1(x)
        l2 = F.relu(l2)
        l2_1 = self._upsample_add(l3_2, l2)
        l2_2 = self.cov2(l2_1)
        l2_2 = F.relu(l2_2)
        l2_3 = self.po3(l2_2)
        l2_4 = l2_3.view(l2_3.size(0), -1)
        o3 = self.fc3(l2_4)

        return o1, o2, o3


def vit_b16_CHR(num_classes, pretrained=True):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    preprocessing = ViT_B_16_Weights.DEFAULT.transforms()
    return ViTb16CHR(model, preprocessing, num_classes)