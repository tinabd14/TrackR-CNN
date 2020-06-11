# Cf. https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/basemodel.py

import tensorflow as tf
from network.Layer import Layer
from network.ConvolutionalLayers import Conv
from network.Util import max_pool, prepare_input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten


def resnet_shortcut(l, n_out, stride, tower_setup):
  n_in = l.get_shape().as_list()[3]
  if n_in != n_out:  # change dimension when channel is not the same
    if stride == 2:
      l = l[:, :-1, :-1, :]
      return Conv(name='convshortcut', inputs=[l], n_features=n_out, tower_setup=tower_setup, filter_size=(1, 1),
               strides=(stride, stride), activation='linear', padding='VALID', batch_norm=True, old_order=True).outputs[0]
    else:
      return Conv(name='convshortcut', inputs=[l], n_features=n_out, tower_setup=tower_setup, filter_size=(1, 1),
               strides=(stride, stride), activation='linear', batch_norm=True, old_order=True).outputs[0]
  else:
    return l


def resnet_bottleneck(l, ch_out, stride, tower_setup):
  l, shortcut = l, l

  l = Conv(name='conv1', inputs=[l], n_features=ch_out, tower_setup=tower_setup, filter_size=(1, 1),
           activation='relu', batch_norm=True, old_order=True).outputs[0]
  if stride == 2:
    l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
    l = Conv(name='conv2', inputs=[l], n_features=ch_out, tower_setup=tower_setup, filter_size=(3, 3), strides=(2, 2),
             activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  else:
    l = Conv(name='conv2', inputs=[l], n_features=ch_out, tower_setup=tower_setup, filter_size=(3, 3),
             strides=(stride, stride), activation='relu', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv3', inputs=[l], n_features=ch_out * 4, tower_setup=tower_setup, filter_size=(1, 1),
           activation='linear', batch_norm=True, old_order=True).outputs[0]
  return l + resnet_shortcut(shortcut, ch_out * 4, stride, tower_setup)


def resnet_group(l, name, features, count, stride, tower_setup):
  with tf.variable_scope(name):
    for i in range(0, count):
      with tf.variable_scope('block{}'.format(i)):
        l = resnet_bottleneck(l, features,
                       stride if i == 0 else 1, tower_setup)
        l = tf.nn.relu(l)
  return l


def pretrained_resnet_conv4(image, num_blocks, tower_setup, freeze_c2=True):
  assert len(num_blocks) == 3
  l = tf.pad(image, [[0, 0], [2, 3], [2, 3], [0, 0]])
  l = Conv(name='conv0', inputs=[l], n_features=64, tower_setup=tower_setup, filter_size=(7, 7), strides=(2, 2),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='VALID')
  c2 = resnet_group(l, 'group0', 64, num_blocks[0], 1, tower_setup)
  if freeze_c2:
    c2 = tf.stop_gradient(c2)
  c3 = resnet_group(c2, 'group1', 128, num_blocks[1], 2, tower_setup)
  #if tower_setup.is_training:
  #  tf.add_to_collection('checkpoints', c3)
  c4 = resnet_group(c3, 'group2', 256, num_blocks[2], 2, tower_setup)
  return [c2, c3, c4]

def pretrained_vgg16_2(image, num_blocks, tower_setup, freeze_c2=True):
  assert len(num_blocks) == 3
  l = tf.pad(image, [[0, 0], [2, 3], [2, 3], [0, 0]])

  l = Conv(name='conv0', inputs=[l], n_features=64, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv1', inputs=[l], n_features=64, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]

  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[2, 2], strides=[2, 2], padding='VALID')

  l = Conv(name='conv2', inputs=[l], n_features=128, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv3', inputs=[l], n_features=128, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]

  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[2, 2], strides=[2, 2], padding='VALID')

  l = Conv(name='conv4', inputs=[l], n_features=256, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv5', inputs=[l], n_features=256, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv6', inputs=[l], n_features=256, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]

  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[2, 2], strides=[2, 2], padding='VALID')

  l = Conv(name='conv7', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv8', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv9', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]

  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[2, 2], strides=[2, 2], padding='VALID')

  l = Conv(name='conv10', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv11', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv12', inputs=[l], n_features=512, tower_setup=tower_setup, filter_size=(7, 7),
           activation='relu', padding='SAME', batch_norm=True, old_order=True).outputs[0]

  l = tf.pad(l, [[0, 0], [0, 1], [0, 1], [0, 0]])
  l = max_pool(l, shape=[2, 2], strides=[2, 2], padding='VALID')
  c2 = resnet_group(l, 'group0', 64, num_blocks[0], 1, tower_setup)
  if freeze_c2:
    c2 = tf.stop_gradient(c2)
  c3 = resnet_group(c2, 'group1', 128, num_blocks[1], 2, tower_setup)
  #if tower_setup.is_training:
  #  tf.add_to_collection('checkpoints', c3)
  c4 = resnet_group(c3, 'group2', 256, num_blocks[2], 2, tower_setup)

  return [c2, c3, c4]



def pretrained_xception_2(image, num_blocks, tower_setup, freeze_c2=True):
  assert len(num_blocks) == 3
  l = tf.pad(image, [[0, 0], [2, 3], [2, 3], [0, 0]])

  l = Conv(name='conv0', inputs=[l], n_features=32, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv1', inputs=[l], n_features=64, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = Conv(name='conv2', inputs=[l], n_features=128, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv3', inputs=[l], n_features=128, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='SAME')

  l = Conv(name='conv4', inputs=[l], n_features=256, tower_setup=tower_setup, filter_size=(3, 3),
         activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv5', inputs=[l], n_features=256, tower_setup=tower_setup, filter_size=(3, 3),
         activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='SAME')

  l = Conv(name='conv6', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv7', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='SAME')


  l = Conv(name='conv8', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv9', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv10', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]



  l = Conv(name='conv11', inputs=[l], n_features=728, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv12', inputs=[l], n_features=1024, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='SAME')

  l = Conv(name='conv13', inputs=[l], n_features=1536, tower_setup=tower_setup, filter_size=(3, 3),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]
  l = Conv(name='conv14', inputs=[l], n_features=2048, tower_setup=tower_setup, filter_size=(3, 3),
         activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]

  l = max_pool(l, shape=[3, 3], strides=[2, 2], padding='SAME')

  l = Conv(name='conv15', inputs=[l], n_features=2048, tower_setup=tower_setup, filter_size=(1, 1),
           activation='relu', padding='VALID', batch_norm=True, old_order=True).outputs[0]


  c2 = resnet_group(l, 'group0', 64, num_blocks[0], 1, tower_setup)
  if freeze_c2:
    c2 = tf.stop_gradient(c2)
  c3 = resnet_group(c2, 'group1', 128, num_blocks[1], 2, tower_setup)
  c4 = resnet_group(c3, 'group2', 256, num_blocks[2], 2, tower_setup)

  return [c2, c3, c4]











def pretrained_resnet50_conv4(image, tower_setup, freeze_c2=True):
  return pretrained_resnet_conv4(image, [3, 4, 6], tower_setup, freeze_c2)


def pretrained_resnet101_conv4(image, tower_setup, freeze_c2=True):
  return pretrained_resnet_conv4(image, [3, 4, 23], tower_setup, freeze_c2)

def pretrained_vgg16(image, tower_setup, freeze_c2=True):
  return pretrained_vgg16_2(image, [3, 4, 23], tower_setup, freeze_c2)

def pretrained_xception(image, tower_setup, freeze_c2=True):
  return pretrained_xception_2(image, [3, 4, 23], tower_setup, freeze_c2)

def add_resnet_conv5(image, tower_setup):
  c5 = resnet_group(image, 'group3', 512, 3, 2, tower_setup)
  return [c5]


# See also savitar1
class ResNet(Layer):
  def __init__(self, name, inputs, tower_setup, variant, add_conv5):
    super(ResNet, self).__init__()
    inp, n_features_inp = prepare_input(inputs)
    # for grayscale
    if n_features_inp == 1:
      inp = tf.concat([inp, inp, inp], axis=-1)
    else:
      assert n_features_inp == 3

    with tf.variable_scope(name):
      if variant == "resnet50":
        net = pretrained_resnet50_conv4(inp, tower_setup)
      elif variant == "resnet101":
        net = pretrained_resnet101_conv4(inp, tower_setup)
      elif variant == "vgg16":
        net = pretrained_vgg16(inp, tower_setup)
      elif variant == "xception":
        net = pretrained_xception(inp, tower_setup)
      else:
        assert False, "Unknown ResNet variant"

      if add_conv5:
        net = add_resnet_conv5(net[-1], tower_setup)

      self.outputs = [net[-1]]
      if tower_setup.is_training:
        vars_to_regularize = tf.trainable_variables(name + "/(?:group1|group2|group3)/.*W")
        regularizers = [1e-4 * tf.nn.l2_loss(W) for W in vars_to_regularize]
        regularization_loss = tf.add_n(regularizers, "regularization_loss")
        self.regularizers.append(regularization_loss)


class ResNet50(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(ResNet50, self).__init__(name, inputs, tower_setup, variant="resnet50", add_conv5=True)


class ResNet50Conv4(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(ResNet50Conv4, self).__init__(name, inputs, tower_setup, variant="resnet50", add_conv5=False)


class ResNet101(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(ResNet101, self).__init__(name, inputs, tower_setup, variant="resnet101", add_conv5=True)


class ResNet101Conv4(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(ResNet101Conv4, self).__init__(name, inputs, tower_setup, variant="resnet101", add_conv5=False)

class VGG_TOM(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(VGG_TOM, self).__init__(name, inputs, tower_setup, variant="vgg16", add_conv5=False)


class XCeption_TOM(ResNet):
  def __init__(self, name, inputs, tower_setup):
    super(XCeption_TOM, self).__init__(name, inputs, tower_setup, variant="xception", add_conv5=False)

