import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


# 子类式构建模型
class VGG_16(tf.keras.Model):
    def __init__(self):
        super(VGG_16, self).__init__()
        # 定义网络结
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv1_1')
        self.bn1_1 = layers.BatchNormalization()
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv1_2')
        self.bn1_2 = layers.BatchNormalization()

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv2_1')
        self.bn2_1 = layers.BatchNormalization()
        self.conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv2_2')
        self.bn2_2 = layers.BatchNormalization()

        # Block 3
        self.conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv3_1')
        self.bn3_1 = layers.BatchNormalization()
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv3_2')
        self.bn3_2 = layers.BatchNormalization()
        self.conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv3_3')
        self.bn3_3 = layers.BatchNormalization()

        # Block 4
        self.conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv4_1')
        self.bn4_1 = layers.BatchNormalization()
        self.conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv4_2')
        self.bn4_2 = layers.BatchNormalization()
        self.conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv4_3')
        self.bn4_3 = layers.BatchNormalization()

        # Block 4
        self.conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv5_1')
        self.bn5_1 = layers.BatchNormalization()
        self.conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv5_2')
        self.bn5_2 = layers.BatchNormalization()
        self.conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                              use_bias=True, activation='relu', name='conv5_3')
        self.bn5_3 = layers.BatchNormalization()

    def call(self, input, training=False):
        # 输入每层的数据
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=input)
        bgr = tf.concat(axis=3, values=[blue - self.VGG_MEAN[0], green - self.VGG_MEAN[1], red - self.VGG_MEAN[2]])
        # Block_1
        conv = self.conv1_1(bgr)
        conv = self.bn1_1(conv, training)
        conv = self.conv1_2(conv)
        conv = self.bn1_2(conv, training)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_1')

        # Block_2
        conv = self.conv2_1(conv)
        conv = self.bn2_1(conv, training)
        conv = self.conv2_2(conv)
        conv = self.bn2_2(conv, training)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_1')
        # Block_3
        conv = self.conv3_1(conv)
        conv = self.bn3_1(conv, training)
        conv = self.conv3_2(conv)
        conv = self.bn3_2(conv, training)
        conv = self.conv3_3(conv)
        conv = self.bn3_3(conv, training)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_1')
        # Block_4
        conv = self.conv4_1(conv)
        conv = self.bn4_1(conv, training)
        conv = self.conv4_2(conv)
        conv = self.bn4_2(conv, training)
        conv = self.conv4_3(conv)
        conv = self.bn4_3(conv, training)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_1')
        # Block_5
        conv = self.conv5_1(conv)
        conv = self.bn5_1(conv, training)
        conv = self.conv5_2(conv)
        conv = self.bn5_2(conv, training)
        conv = self.conv5_3(conv)
        conv = self.bn5_3(conv, training)
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5_1')

        return conv


def load_model(pretrain_path, model):
    weighs = np.load(pretrain_path, encoding='latin1').item()
    for layer_name in weighs.keys():
        if 'conv' in layer_name:  # 只加载卷积层
            print('layer_name:', layer_name)
            layer = model.get_layer(layer_name)
            # print(layer)
            layer.set_weights(weighs[layer_name])

if __name__ == '__main__':
    # bulid model
    model = VGG_16()
    input_layer = tf.keras.layers.Input([224, 224, 3])
    model(input_layer)
    # load pretrain model
    pretrain_path = 'vgg16.npy'
    load_model(pretrain_path, model)
    for var in model.variables:
        print(var.name)
    #run model
    inputs = tf.random.normal(shape=[1, 224, 224, 3])
    out = model(inputs)
    print(out)
