import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        
import SCCAN_PAHNet.model.resnet as models
from SCCAN_PAHNet.model.PPM import PPM

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.pretrained = True
        self.classes = 16 if self.dataset=='pascal' else 61
        
        assert self.layers in [50, 101, 152]

        print('INFO: Using ResNet {}'.format(self.layers))
        if self.layers == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.layers == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        else:
            resnet = models.resnet152(pretrained=self.pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # Base Learner
        self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
        fea_dim = 512 if self.vgg else 2048
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [     
            {'params': model.encoder.parameters()},
            {'params': model.ppm.parameters()},
            {'params': model.cls.parameters()},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer


    def forward(self, x, y_m):
        x_size = x.size()
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)    # 473

        x = self.encoder(x)
        x = self.ppm(x)
        x = self.cls(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(x, y_m.long())
            return x.max(1)[1], main_loss
        else:
            return x

