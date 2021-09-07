import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


def build_model(config, num_class=100):
    model_name = config.model
    if model_name == "mobv2":
        model = models.mobilenet_v2(pretrained=False)
        in_feature = 1280
        model.classifier = nn.Linear(in_feature, num_class)
    elif model_name == "mobv3_small":
        model = models.mobilenet_v3_small(pretrained=False)
        in_feature = 576
        model.classifier = nn.Linear(in_feature, num_class)
    elif model_name == "mobv3_large":
        model = models.mobilenet_v3_large(pretrained=False)
        in_feature = 960
        model.classifier = nn.Linear(in_feature, num_class)
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=num_class, bias=True),
        )
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=num_class, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=4096, out_features=num_class, bias=True)
        )
    elif model_name == "efficientnet":
        model_type = model_name + config.model_type
        model = EfficientNet.from_name(model_type, num_classes=num_class)

    return model
