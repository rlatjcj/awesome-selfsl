import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import data_utils

from typing import Optional, Tuple, Callable


BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101':
        ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
}


def ResNet(
    stack_fn: Callable[[tf.Tensor], tf.Tensor],
    preact: bool,
    model_name: str = 'resnet',
    include_top: bool = True,
    weights: bool = 'imagenet',
    input_shape: Optional[Tuple[int]] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    classifier_activation: str = 'softmax',
    first_conv: bool = True,
    maxpool1: bool = True,
    **kwargs
) -> tf.keras.Model:

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    img_input = layers.Input(shape=input_shape)

    if first_conv:
        x = layers.ZeroPadding2D(
            padding=((1, 1), (1, 1)), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, 3, strides=1, use_bias=False, name='conv1_conv')(x)
    else:
        x = layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    if maxpool1:
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    else:
        x = layers.MaxPooling2D(1, strides=1, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = tf.keras.Model(img_input, x, name=model_name)

    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block0(
    x: tf.Tensor, 
    filters: int, 
    kernel_size: int = 3, 
    stride: int = 1, 
    conv_shortcut: bool = True, 
    name: Optional[str] = None
) -> tf.Tensor:

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, 
            use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
            filters, kernel_size, strides=stride, 
            use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
            filters, kernel_size, padding='SAME', 
            use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack0(
    x: tf.Tensor, 
    filters: int, 
    blocks: int, 
    stride1: int = 2, 
    name: Optional[str] = None
) -> tf.Tensor:

    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def block1(
    x: tf.Tensor, 
    filters: int, 
    kernel_size: int = 3, 
    stride: int = 1, 
    conv_shortcut: bool = True, 
    name: Optional[str] = None
) -> tf.Tensor:

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, 
            use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
            filters, 1, strides=stride, 
            use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
            filters, kernel_size, padding='SAME', 
            use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
            4 * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(
    x: tf.Tensor, 
    filters: int, 
    blocks: int, 
    stride1: int = 2, 
    name: Optional[str] = None
) -> tf.Tensor:

    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet18(
    include_top: bool = True,
    weights: str = 'imagenet',
    input_shape: Tuple[int] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    **kwargs
) -> tf.keras.Model:
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=1, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        return stack0(x, 512, 2, name='conv5')

    return ResNet(stack_fn, False, 'resnet18', include_top, weights,
                  input_shape, pooling, classes, **kwargs)


def ResNet34(
    include_top: bool = True,
    weights: str = 'imagenet',
    input_shape: Tuple[int] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    **kwargs
) -> tf.keras.Model:
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack0(x, 64, 3, stride1=1, name='conv2')
        x = stack0(x, 128, 4, name='conv3')
        x = stack0(x, 256, 6, name='conv4')
        return stack0(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, 'resnet34', include_top, weights,
                  input_shape, pooling, classes, **kwargs)


def ResNet50(
    include_top: bool = True,
    weights: str = 'imagenet',
    input_shape: Tuple[int] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    **kwargs
) -> tf.keras.Model:
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, 'resnet50', include_top, weights,
                  input_shape, pooling, classes, **kwargs)


def ResNet101(
    include_top: bool = True,
    weights: str = 'imagenet',
    input_shape: Tuple[int] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    **kwargs
) -> tf.keras.Model:
    """Instantiates the ResNet101 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, 'resnet101', include_top, weights,
                  input_shape, pooling, classes, **kwargs)


def ResNet152(
    include_top: bool = True,
    weights: str = 'imagenet',
    input_shape: Tuple[int] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    **kwargs
) -> tf.keras.Model:
    """Instantiates the ResNet152 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, 'resnet152', include_top, weights,
                  input_shape, pooling, classes, **kwargs)