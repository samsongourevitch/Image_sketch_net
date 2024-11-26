import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import timm
import lightly
import os

nclasses = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
class Resnet101_based(nn.Module):
    def __init__(self):
        super(Resnet101_based, self).__init__()
        
        # Load a pre-trained ResNet-101 model
        self.model = models.resnet101(weights='ResNet101_Weights.DEFAULT')

        # for param in list(self.model.parameters())[:-2]:
        #     param.requires_grad = False
        
        # Replace the final fully connected layer for the specific number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)
        
    def forward(self, x):
        return self.model(x)
    
class SketchClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(SketchClassifier, self).__init__()
        # Define classifier layers based on computed feature dimension
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)   # Dropout for regularization
        self.fc2 = nn.Linear(512, nclasses)  # Output layer for class prediction

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class ViT_based(nn.Module):
    def __init__(self):
        super(ViT_based, self).__init__()
        
        # Load a pre-trained Vision Transformer model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        
        # Replace the final fully connected layer for the specific number of classes
        self.model.head = nn.Linear(self.model.head.in_features, nclasses)
        
    def forward(self, x):
        return self.model(x)
    
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        self.model = lightly.models.SimCLR(pretrained=True, base_model='resnet-50')
        self.model.fc = nn.Linear(self.model.fc.in_features, nclasses)

    def forward(self, x):
        with torch.no_grad():
            h = self.model(x)
        return self.fc(h)
    
class EfficientNet_based(nn.Module):
    def __init__(self):
        super(EfficientNet_based, self).__init__()
        
        # Load a pre-trained EfficientNet model
        self.model = models.efficientnet_b7(pretrained=True)

        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, nclasses)
        
    def forward(self, x):
        return self.model(x)
    
class MetaModel(nn.Module):
    def __init__(self, load_models):
        super(MetaModel, self).__init__()
        self.models = []
        self.model_1 = Resnet_based()
        self.model_2 = Resnet101_based()

        self.model_1.load_state_dict(torch.load(load_models[0]))
        self.model_2.load_state_dict(torch.load(load_models[1]))

        self.models = [self.model_1, self.model_2]
        
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        
        self.fc = nn.Linear(nclasses*len(self.models), nclasses)
        
    def forward(self, x):
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                model.to(device)
                outputs.append(model(x))
            stacked_outputs = torch.stack(outputs, dim=2)
        meta_input = stacked_outputs.view(stacked_outputs.size(0), -1)
        return self.fc(meta_input)
    

