#from convNet import ConvNet
#model = ConvNet(training=False)
#model.restore_session()
#pred = model.predict('./test_img.jpg')
#print(pred)

import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imresize

path_graph_def = './graphs/tinyNet.pb'

graph = tf.Graph()
with graph.as_default():
    with tf.gfile.FastGFile(path_graph_def, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')
    inputs = graph.get_tensor_by_name('x:0')

    
out = graph.get_tensor_by_name("Softmax:0")
session = tf.InteractiveSession(graph=graph)

img = imread('./men.jpg')
img = imresize(img, [48, 48])
img = np.expand_dims(img, axis=0)

feed_dict = {inputs: img}
values = session.run(out, feed_dict=feed_dict)
values = np.squeeze(values)
print()
print()
print(values)
#if values == 0:
#    print("girl")
#elif values == 1:
#    print("boy")
#else:
#    print("Error")
