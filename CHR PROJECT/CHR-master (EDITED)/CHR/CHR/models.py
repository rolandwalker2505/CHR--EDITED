import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTb16CHR(nn.Module):
    def __init__(self, model, num_classes, dense=True):
        super(ViTb16CHR, self).__init__()

        # Use ViT model
        self.model = model
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # Define custom layers for additional processing
        self.cov4 = nn.Conv2d(1280, 1280, kernel_size=1, stride=1)
        self.cov3 = nn.Conv2d(1280, 1024, kernel_size=1, stride=1)
        self.cov2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)

        self.cov3_1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.cov2_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.po1 = nn.AvgPool2d(7, stride=1)
        self.po2 = nn.AvgPool2d(14, stride=1)
        self.po3 = nn.AvgPool2d(28, stride=1)
        self.fc1 = nn.Linear(1280, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc3 = nn.Linear(512, num_classes)


    def _upsample_add(self,x,y):
        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)
        return torch.cat([z,y],1)
    def get_config_optim(self, lr, lrp):
        return [{'params': self.model.parameters()},
                {'params': self.fc.parameters()},
                {'params': self.cov4.parameters()},
                {'params': self.cov3.parameters()},
                {'params': self.cov2.parameters()},
                {'params': self.cov3_1.parameters()},
                {'params': self.cov2_1.parameters()},
                {'params': self.po1.parameters()},
                {'params': self.po2.parameters()},
                {'params': self.po3.parameters()},
                {'params': self.fc1.parameters()},
                {'params': self.fc2.parameters()},
                {'params': self.fc3.parameters()}]

    def forward(self, x):
        # Extract features from ViT
        x = self.model.forward_features(x)  # ViT model feature extraction
        x = x.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions [B, C] -> [B, C, 1, 1]

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
        return o1,o2,o3

def vit_b16_CHR(num_classes, pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    return ViTb16CHR(model, num_classes )

