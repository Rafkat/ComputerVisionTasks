from convolution_models.AlexNet import AlexNet
from convolution_models.ConvNext import ConvNext
from convolution_models.GoogLeNet import GoogLeNet
from convolution_models.LeNet5 import LeNet5
from convolution_models.MobileNet_v1 import MobileNetV1
from convolution_models.MobileNet_v2 import MobileNetV2
from convolution_models.ResNet34 import ResNet34
from convolution_models.ResNet50 import ResNet50
from convolution_models.ResNext import ResNeXt
from convolution_models.VGG16 import VGG16
from convolution_models.Xception import Xception

VGG16_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': VGG16(input_size=32, nb_classes=10),
}

ALEXNET_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': AlexNet(input_size=32, nb_classes=10),
}

CONVNEXT_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': ConvNext(nb_classes=10)
}

GOOGLENET_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': GoogLeNet(nb_classes=10),
}

LENET_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': LeNet5(in_channels=3, nb_classes=10),
}

MOBILENETV1_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': MobileNetV1(nb_classes=10),
}

MOBILENETV2_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': MobileNetV2(nb_classes=10),
}

RESNET34_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': ResNet34(nb_classes=10),
}

RESNET50_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': ResNet50(nb_classes=10),
}

RESNEXT_CONFIG = {
    'batch_size': 16,
    'img_size': 128,
    'model': ResNeXt(nb_classes=10),
}

XCEPTION_CONFIG = {
    'batch_size': 64,
    'img_size': 224,
    'model': Xception(nb_classes=10),
}


models_config = {
    'vgg16': VGG16_CONFIG,
    'alexnet': ALEXNET_CONFIG,
    'convnext': CONVNEXT_CONFIG,
    'googlenet': GOOGLENET_CONFIG,
    'lenet': LENET_CONFIG,
    'mobilenetv1': MOBILENETV1_CONFIG,
    'mobilenetv2': MOBILENETV2_CONFIG,
    'resnet34': RESNET34_CONFIG,
    'resnet50': RESNET50_CONFIG,
    'resnext': RESNEXT_CONFIG,
    'xception': XCEPTION_CONFIG,
}