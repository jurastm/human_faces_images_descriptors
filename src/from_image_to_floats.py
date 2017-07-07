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
import argparse

def get_filepath(filepath):
    count = 0
    new_path = ''
    for i in reversed(filepath):
        if i == '/':
            count += 1
            if count == 2:
                break
        new_path += i
        
    new_path = new_path[::-1]
    
    return new_path[:-4]
    
def process_image(img_path, save_dir="./floating_signatures/", out_layer="fc_1:0"):
    path_graph_def = './graphs/tinyNet.pb'
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(path_graph_def, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')
        inputs = graph.get_tensor_by_name('x:0')
        if out_layer is not None:
            out = graph.get_tensor_by_name(out_layer)
        else:
            out = graph.get_tensor_by_name("fc_1:0")
        
    session = tf.InteractiveSession(graph=graph)

    img = imread(img_path)
    img = imresize(img, [48, 48])

    feed_dict = {inputs: [img]}
    values = session.run(out, feed_dict=feed_dict)
    values = np.squeeze(values)
    
    filepath = get_filepath(img_path)
    if save_dir is not None:
        save_path = os.path.join(save_dir, filepath)
    else:
        save_path = os.path.join("./floating_signatures/", filepath)
    np.save(save_path, values)
  
    return values
    
def get_e_and_b_signature(floats, out_layer="binary:0"):
    path_graph_def = './graphs/binaryGraph.pb'
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(path_graph_def, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')
        inputs = graph.get_tensor_by_name('x:0')
        if out_layer is not None:
            out = graph.get_tensor_by_name(out_layer)
        else:
            out = graph.get_tensor_by_name("binary:0")
    
    session = tf.InteractiveSession(graph=graph)
    tanh_signature = session.run(out, feed_dict={inputs: [floats]})
    tanh_signature = np.squeeze(tanh_signature)
    
    signature = np.ceil(tanh_signature)
    signature = signature.astype('int32')
    binary_signature = signature.astype('bool')
    
    return tanh_signature, binary_signature
    
    
    

if __name__ == "__main__":
    
    desc = "Feed target image into neural network " \
           "and retrieve intermediate values from target layer. "
        
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--img_path", required=True,
                        help="input pathway to image location.")
    
    parser.add_argument("--save_dir", required=False,
                        help="specify directory for saving target layer values")
    
    parser.add_argument("--out_layer", required=False,
                        help="optionally specify desired output layer name")
    
    args = parser.parse_args()
    
    img_path = args.img_path
    save_dir = args.save_dir
    out_layer = args.out_layer
    
    
    floats = process_image(img_path=img_path, save_dir=save_dir, out_layer=out_layer)
    
    
    
    

