import models.classification_convolution as conv

VGG16_CONFIG = {
    'batch_size': 128,
    'img_size': 32,
    'model': conv.VGG16.VGG16(input_size=32, nb_classes=10),
}

ALEXNET_CONFIG = {
    'batch_size': 128,
    'img_size': 32,
    'model': conv.AlexNet.AlexNet(input_size=32, nb_classes=10),
}

CONVNEXT_CONFIG = {
    'batch_size': 128,
    'img_size': 32,
    'model': conv.ConvNext.ConvNext(nb_classes=10)
}

GOOGLENET_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.GoogLeNet.GoogLeNet(nb_classes=10),
}

LENET_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.LeNet5.LeNet5(in_channels=3, nb_classes=10),
}

MOBILENETV1_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.MobileNet_v1.MobileNetV1(nb_classes=10),
}

MOBILENETV2_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.MobileNet_v2.MobileNetV2(nb_classes=10),
}

RESNET34_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.ResNet34.ResNet34(nb_classes=10),
}

RESNET50_CONFIG = {
    'batch_size': 64,
    'img_size': 224,
    'model': conv.ResNet50.ResNet50(nb_classes=10),
}

RESNEXT_CONFIG = {
    'batch_size': 16,
    'img_size': 128,
    'model': conv.ResNext.ResNeXt(nb_classes=10),
}

XCEPTION_CONFIG = {
    'batch_size': 64,
    'img_size': 224,
    'model': conv.Xception.Xception(nb_classes=10),
}

DENSENET_CONFIG = {
    'batch_size': 128,
    'img_size': 224,
    'model': conv.DenseNet.DenseNet(nb_classes=10),
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
#    'resnext': RESNEXT_CONFIG,
#    'xception': XCEPTION_CONFIG,
#    'densenet': DENSENET_CONFIG
}
