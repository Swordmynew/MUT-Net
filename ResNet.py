import torch
from torchvision import models

class CustomResNet50(torch.nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        
        #self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def forward(self, x):
    
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1_output = self.resnet.layer1(x)
        layer2_output = self.resnet.layer2(layer1_output)
        layer3_output = self.resnet.layer3(layer2_output)
        layer4_output = self.resnet.layer4(layer3_output)

        return [layer1_output, layer2_output, layer3_output, layer4_output]

Resnet50 = CustomResNet50()
