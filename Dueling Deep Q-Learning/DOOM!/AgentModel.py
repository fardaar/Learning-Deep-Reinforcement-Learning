import tensorflow as tf
import numpy as np


class AgentModel(tf.keras.Model):
    # Here we implement the agent's neural model
    def __init__(self, input_shape):
        super(AgentModel, self).__init__()
        # input_layer: :|
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, name='Input_Layer')
        # conv_layer1: first convolutional layer
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=32,
                                                  kernel_size=[8, 8],
                                                  strides=[4, 4],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                                  name='Conv_Layer_1')
        # conv_layer1_batchnorm: batchnormalization on conv_layer1's outputs
        self.conv_layer1_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                                        trainable=True,
                                                                        name='Conv_Layer_1_Batchnorm')
        # conv_layer2: second convolutional layer
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=[4, 4],
                                                  strides=[2, 2],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                                  name='Conv_Layer_2')
        # conv_layer2_batchnorm: batchnormalization on conv_layer2's outputs
        self.conv_layer2_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                                        trainable=True,
                                                                        name='Conv_Layer_2_Batchnorm')
        # conv_layer3: third convolutional layer
        self.conv_layer3 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=[4, 4],
                                                  strides=[2, 2],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                                  name='Conv_Layer_3')
        # conv_layer3_batchnorm: batchnormalization on conv_layer3's outputs
        self.conv_layer3_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                                        trainable=True,
                                                                        name='Conv_Layer_3_Batchnorm')

        # flattening the last layers outputs. We do this so we can use them as inputs to a dense layer
        self.flatten = tf.keras.layers.Flatten()
        # a fully connected layer which is going to calculate state's value
        self.value_fc = tf.keras.layers.Dense(units=512,
                                              activation='relu',
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              name='Value_Estimator')
        # calculate the value from value_fc outputs
        self.value = tf.keras.layers.Dense(units=1,
                                           activation=None,
                                           kernel_initializer=tf.keras.initializers.he_normal(),
                                           name='Value_value')
        # a fully connected layer which is going to calculate the action taken in that state's advantage
        self.advantage_fc = tf.keras.layers.Dense(units=512,
                                                  activation='relu',
                                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                                  name='Advantage_Estimator')
        # calculate the advantage from advantage_fc
        self.advantage = tf.keras.layers.Dense(units=1,
                                               activation=None,
                                               kernel_initializer=tf.keras.initializers.he_normal(),
                                               name='Advantage_value')

    @tf.function
    def call(self, inputs):
        # this function implements a forward pass through agent's network
        out = self.input_layer(inputs)
        out = self.conv_layer1(out)
        out = self.conv_layer1_batchnorm(out)
        out = tf.nn.elu(out)
        out = self.conv_layer2(out)
        out = self.conv_layer2_batchnorm(out)
        out = tf.nn.elu(out)
        out = self.conv_layer3(out)
        out = self.conv_layer3_batchnorm(out)
        out = tf.nn.elu(out)
        out = self.flatten(out)
        out1 = self.value_fc(out)
        out2 = self.advantage_fc(out)
        out1_value = self.value(out1)
        out2_value = self.advantage(out2)
        out = out1_value + tf.subtract(out2_value, tf.reduce_mean(out2_value, axis=1, keepdims=True))
        return out


