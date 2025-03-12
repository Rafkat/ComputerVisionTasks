import sys
sys.path.append('../convolution_models')
print(sys.path)
import convolution_models as cm

VGG16_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': cm.VGG16.VGG16(input_size=32, nb_classes=10),
}

ALEXNET_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': cm.AlexNet.AlexNet(input_size=32, nb_classes=10),
}

CONVNEXT_CONFIG = {
    'batch_size': 1024,
    'img_size': 32,
    'model': cm.ConvNext.ConvNext(nb_classes=10)
}

GOOGLENET_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': cm.GoogLeNet.GoogLeNet(nb_classes=10),
}

LENET_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': cm.LeNet5.LeNet5(in_channels=3, nb_classes=10),
}

MOBILENETV1_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': cm.MobileNet_v1.MobileNetV1(nb_classes=10),
}

MOBILENETV2_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': cm.MobileNet_v2.MobileNetV2(nb_classes=10),
}

RESNET34_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': cm.ResNet34.ResNet34(nb_classes=10),
}

RESNET50_CONFIG = {
    'batch_size': 256,
    'img_size': 224,
    'model': cm.ResNet50.ResNet50(nb_classes=10),
}

RESNEXT_CONFIG = {
    'batch_size': 16,
    'img_size': 128,
    'model': cm.ResNext.ResNeXt(nb_classes=10),
}

XCEPTION_CONFIG = {
    'batch_size': 64,
    'img_size': 224,
    'model': cm.Xception.Xception(nb_classes=10),
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
