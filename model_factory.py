"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet
from model import Net, Resnet_based, SketchClassifier

class ModelFactory:
    def __init__(self, model_name, feature_extractor_path: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()
        self.feature_extractor_path = feature_extractor_path

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet_based":
            return Resnet_based()
        elif self.model_name == "sketch_classifier":
            return SketchClassifier(self.feature_extractor_path)
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
