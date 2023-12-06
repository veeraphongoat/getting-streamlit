import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class YourModelClass(nn.Module):
    def __init__(self, num_classes=11):
        super(YourModelClass, self).__init__()

        # Load the MobileNetV3 large model without pretraining
        self.mobilenetv3 = mobilenet_v3_large(pretrained=False)

        # Adjust the last fully connected layer to match the number of classes
        in_features = self.mobilenetv3.classifier[3].in_features
        self.mobilenetv3.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Forward pass through the MobileNetV3 model
        x = self.mobilenetv3(x)
        return x
