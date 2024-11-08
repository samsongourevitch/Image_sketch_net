import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class Resnet_based(nn.Module):
    def __init__(self):
        super(Resnet_based, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # for param in list(self.model.parameters())[:-2]:
        #     param.requires_grad = False
        
        # Replace the final fully connected layer for the specific number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)
        
    def forward(self, x):
        return self.model(x)
    
class SketchClassifier(nn.Module):
    def __init__(self, feature_extractor_path):
        super(SketchClassifier, self).__init__()
        self.feature_extractor = torch.load(feature_extractor_path)
        # Our classifier model
        self.fc1 = nn.Linear(2048, 512)  # Input size is 2048 from ResNet-50's feature map
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)   # Dropout for regularization
        self.fc2 = nn.Linear(512, nclasses)  # Output layer for class prediction

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map from ResNet
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x