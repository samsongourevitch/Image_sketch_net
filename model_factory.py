"""Python file to instantiate the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet, data_transforms_vit, data_transforms_efficientnet
from model import Net, Resnet_based, SketchClassifier, ViT_based, SimCLR, Resnet101_based, EfficientNet_based, MetaModel
import torch

class ModelFactory:
    def __init__(self, model_name, feature_extractor_path=None, load_models=None, use_cuda=True):
        self.model_name = model_name
        self.load_models = load_models
        self.feature_extractor_path = feature_extractor_path
        self.use_cuda = use_cuda
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet_based":
            return Resnet_based()
        elif self.model_name == "resnet101_based":
            return Resnet101_based()
        elif self.model_name == "sketch_classifier":
            return SketchClassifier(2048)
        elif self.model_name == "vit_based":
            return ViT_based()
        elif self.model_name == "simclr":
            return SimCLR()
        elif self.model_name == "efficientnet_based":
            return EfficientNet_based()
        elif self.model_name == "meta_model":
            return MetaModel(self.load_models)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "resnet_based" or self.model_name == "resnet101_based":
            return data_transforms_resnet
        elif self.model_name == "sketch_classifier":
            return data_transforms_resnet
        elif self.model_name == "vit_based":
            return data_transforms_vit
        elif self.model_name == "simclr":
            return data_transforms_resnet
        elif self.model_name == "efficientnet_based":
            return data_transforms_efficientnet
        elif self.model_name == "meta_model":
            return data_transforms_resnet
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
