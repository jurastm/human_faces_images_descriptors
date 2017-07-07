import tensorflow as tf
slim = tf.contrib.slim


def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv_layer(inputs, W_shape, b_shape, padding='SAME'):
    W = weights_variable(shape=W_shape)
    b = bias_variable([b_shape])
    layer = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding=padding)
    layer = tf.nn.bias_add(layer, b)
    
    # layer = tf.nn.relu(layer, name=name)
    return layer

def batch_norm(inputs):
    layer = slim.batch_norm(inputs)
    return layer

def max_pool(inputs, name=None, padding='SAME'):
    layer = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name=name)
    return layer

def avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=None):
    layer = tf.nn.avg_pool(inputs, ksize=ksize, strides=strides, name=name)
    return layer

def flatten_layer(inputs):
    layer_shape = inputs.get_shape()
    num_features = layer_shape[1:].num_elements()
    layer_flat = tf.reshape(inputs, [-1, num_features])
    return layer_flat, num_features

def fc_layer(inputs, num_inputs, num_outputs, name=None,  use_relu=True):
    W = weights_variable([num_inputs, num_outputs])
    b = bias_variable([num_outputs])
    layer = tf.matmul(inputs, W)
    layer = tf.nn.bias_add(layer, b)
    if use_relu:
        layer = tf.nn.relu(layer, name=name)
    return layer

def binary_layer(inputs, num_inputs, num_outputs, name='binary_layer'):
    W = weights_variable([num_inputs, num_outputs])
    b = bias_variable([num_outputs])
    layer = tf.matmul(inputs, W)
    layer = tf.nn.bias_add(layer, b)
    
    layer = tf.nn.tanh(layer, name=name)
    return layer
