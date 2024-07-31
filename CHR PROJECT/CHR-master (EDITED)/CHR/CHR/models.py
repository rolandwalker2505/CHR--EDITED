import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
class EfficientNetB0CHR(nn.Module):
    def __init__(self, model, num_classes, dense=True):
        super(EfficientNetB0CHR, self).__init__()
        #self.dense = dense
        for item in model.children():
            if isinstance(item, nn.BatchNorm2d):
                item.affine = False

        self.features = nn.Sequential(
           model.features,
           model.avg_pooling,
        )
        self.fc = nn.Linear(model.fc.in_features, num_classes)

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

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # self.dropout = nn.Dropout(p=0.5)
        #
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def _upsample_add(self,x,y):

        _, _, H, W = y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return torch.cat([z,y],1)
    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters()},
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
        x = self.features(x)
        x = self.fc(x)

        l4 = self.cov4(x)
        l4 = F.relu(l4)
        l4_1 = self.po1(l4)
        l4_2 = l4_1.view(l4_1.size(0), -1)
        o1 = self.fc1(l4_2)

        l3 = self.cov3_1(x)
        l3 = F.relu(l3)
        l3_1 = self._upsample_add(l4,l3)
        l3_2 = self.cov3(l3_1)
        l3_2 = F.relu(l3_2)
        l3_3 = self.po2(l3_2)
        l3_4 = l3_3.view(l3_3.size(0), -1)
        o2 = self.fc2(l3_4)

        l2 = self.cov2_1(x)
        l2 = F.relu(l2)
        l2_1 = self._upsample_add(l3_2,l2)
        l2_2 = self.cov2(l2_1)
        l2_2 = F.relu(l2_2)
        l2_3 = self.po3(l2_2)
        l2_4 = l2_3.view(l2_3.size(0), -1)
        o3 = self.fc3(l2_4)

        return o1,o2,o3

def efficientnet_CHR(num_classes, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)

    return EfficientNetB0CHR(model, num_classes )

