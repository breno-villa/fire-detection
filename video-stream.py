import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'C:/Tensorflow/feito/inceptionColor/output_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('C:\Tensorflow\models\research\object_detection\data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = 'C:/Tensorflow/feito/inceptionColor/output_labels.txt'




# ## Download Model

# In[5]:




# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

#detection_graph = tf.Graph()
#with detection_graph.as_default():
#  od_graph_def = tf.GraphDef()
#  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#    serialized_graph = fid.read()
#    od_graph_def.ParseFromString(serialized_graph)
#    tf.import_graph_def(od_graph_def, name='')


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#print(label_map)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:




def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.


# Size, in inches, of the output images.



# In[10]:

input_layer = "Placeholder"
output_layer = "final_result"

input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_mean = 0
input_std = 255
input_height = 299
input_width = 299

file_name = "C:/Tensorflow/teste2.jpg"

graph = load_graph(PATH_TO_CKPT)

resize=(299,299)
currentFrame = 0
labels = load_labels(PATH_TO_LABELS)


with graph.as_default():
  with tf.Session(graph=graph) as sess:
    while True:
      ret, image_np = cap.read()

      currentFrame += 1
      name = 'C:/Tensorflow/dataTemp/' + str(currentFrame) + '.jpg'
      cv2.imwrite(name, image_np)
      file_name = name
      #if currentFrame % 2 == 0:
        #file_name = "C:/Tensorflow/teste.jpg"
      #else:
        #file_name = "C:/Tensorflow/teste2.jpg"

      #print(file_name)

      if currentFrame != 0:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np = cv2.resize(image_np,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
        #np_image_data = np.asarray(image_np)
        #np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        #image_np_expanded = np.expand_dims(np_image_data, axis=0)
        

        t = read_tensor_from_image_file(
              file_name,
              input_height=input_height,
              input_width=input_width,
              input_mean=input_mean,
              input_std=input_std)

        
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        results = sess.run(output_operation.outputs[0], 
                  {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        
        for i in top_k:
              print(labels[i], results[i])
      #image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      #boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      #scores = detection_graph.get_tensor_by_name('detection_scores:0')
      #classes = detection_graph.get_tensor_by_name('detection_classes:0')
      #num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      #(boxes, scores, classes, num_detections) = sess.run(
          #[boxes, scores, classes, num_detections],
          #feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      #vis_util.visualize_boxes_and_labels_on_image_array(
          #image_np,
          #np.squeeze(boxes),
          #np.squeeze(classes).astype(np.int32),
          #np.squeeze(scores),
          #category_index,
          #use_normalized_coordinates=True,
          #line_thickness=8)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
