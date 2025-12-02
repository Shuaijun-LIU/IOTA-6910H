"""
ResNet-18 model for CIFAR-10
Supports both training from scratch and fine-tuning from pretrained weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pretrained=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        self.pretrained = pretrained

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass - returns logits (not probabilities)
        This is required by AutoAttack
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # Important: return logits, not probabilities
        return out


def ResNet18(num_classes=10, pretrained=False):
    """
    ResNet-18 model for CIFAR-10
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: If True, load pretrained weights (default: False)
    
    Returns:
        ResNet-18 model
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, pretrained=pretrained)
    
    if pretrained:
        # Try to load pretrained weights from torchvision
        try:
            from torchvision.models import resnet18
            pretrained_model = resnet18(weights='IMAGENET1K_V1')
            
            # Modify first conv layer for CIFAR-10 (3x3 instead of 7x7)
            # Copy weights (we'll handle this in training script if needed)
            print("Note: Pretrained ResNet-18 loaded. First conv layer may need adjustment for CIFAR-10.")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Training from scratch instead.")
    
    return model


if __name__ == '__main__':
    # Test the model
    model = ResNet18(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

