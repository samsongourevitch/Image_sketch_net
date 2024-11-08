"""Python file to instantiate the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet
from model import Net, Resnet_based, SketchClassifier
from torchvision import models
import torch

class ModelFactory:
    def __init__(self, model_name, feature_extractor_path=None):
        self.model_name = model_name
        self.feature_extractor_path = feature_extractor_path
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet_based":
            return Resnet_based()
        elif self.model_name == "sketch_classifier":
            # Load the pre-trained ResNet-50 model structure
            resnet = models.resnet50()
            resnet.load_state_dict(torch.load(self.feature_extractor_path))
            # Remove the last layer
            feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            return SketchClassifier(feature_extractor)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "basic_cnn2":
            return data_transforms
        elif self.model_name == "resnet_based":
            return data_transforms_resnet
        elif self.model_name == "sketch_classifier":
            return data_transforms_resnet
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
