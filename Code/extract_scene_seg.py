# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given a list of images, run scene semantic segmentation using deeplab."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
# import tensorflow.contrib.slim as slim
# def model_summary():
#     # model_vars = tf.compat.v1.model_variables()
#     # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
#     slim.model_analyzer.analyze_ops(tf.compat.v1.get_default_graph(),print_info=True)

def resize_seg_map(seg, down_rate, keep_full=False):
  img_ = Image.fromarray(seg.astype(dtype=np.uint8))
  w_, h_ = img_.size
  neww, newh = int(w_ / down_rate), int(h_ / down_rate)
  if keep_full:
    neww, newh = 512, 288

  newimg = img_.resize((neww, newh))  # neareast neighbor

  newdata = np.array(newimg)
  return newdata

def extract_scene_seg(dataset_resize,model_path,every =1,down_rate=8.0,job= 1,curJob =1,gpuid= 0, keep_full=False):
  seg_output={}
  input_size = 513  # the model's input size, has to be this
  dataset_resize={i:dataset_resize[i] for i in range(0,len(dataset_resize),every)}
  # load the model graph
  print("loading model...")
  graph = tf.Graph()
  with graph.as_default():
    gd = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, "rb") as f:
      sg = f.read()
      gd.ParseFromString(sg)
      tf.import_graph_def(gd, name="")

    input_tensor = graph.get_tensor_by_name("ImageTensor:0")
    output_tensor = graph.get_tensor_by_name("SemanticPredictions:0")

  print("loaded.")
  
  #Configration
  tfconfig = tf.compat.v1.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [gpuid]]))

  with graph.as_default():
    with tf.compat.v1.Session(graph=graph, config=tfconfig) as sess:
      count = 0  
      for frame_ind , img in tqdm(dataset_resize.items()):
        count += 1
        if (count % job) != (curJob - 1):
          continue

        #resize to meet model's input size  
        h, w, _ = img.shape
        resize_r = 1.0 * input_size / max(w, h)
        target_size = (int(resize_r * w), int(resize_r * h))
        resize_img= cv2.resize(img, target_size,  cv2.INTER_NEAREST )
        

        #Run model and get seg_map for each frame 
        seg_map, = sess.run([output_tensor],
                            feed_dict={input_tensor: [np.asarray(resize_img)]})

        seg_map = seg_map[0]  # single image input test

        seg_map = resize_seg_map(seg_map, down_rate,keep_full)
        seg_output[frame_ind]=seg_map
        
    # model_summary()

  return seg_output






