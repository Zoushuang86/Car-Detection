import argparse
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append('G:/tensorflow1.5/models/research')
sys.path.append('G:/tensorflow1.5/models/research/slim')
sys.path.append('G:/tensorflow1.5/models/research/object_detection')
from utils import visualization_utils as vis_util
from utils import label_map_util
import my_classify_image as classifier
#if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!',tf.__version__)

NUM_CLASSES = 20


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'exported_graphs/frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'pascal_label_map.pbtxt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    test_img_path = os.path.join(FLAGS.dataset_dir, 'test.jpg')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(test_img_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            # Choose the score more than 0.5 and the class's id is 7
            d_scores = []
            d_classes = []
            d_boxes = []
            d_images = []
            img_height = image_np.shape[0]
            img_width = image_np.shape[1]
            num = boxes.shape[0]
            for i in range(num):
              if classes[i] == 7 and scores[i] >= 0.5:
                d_boxes.append(boxes[i])
                d_classes.append(classes[i])
                d_scores.append(scores[i])
                img_scrop = image_np[int(boxes[i][0]*img_height):int(boxes[i][2]*img_height), int(boxes[i][1]*img_width):int(boxes[i][3]*img_width), :]
                d_images.append(img_scrop)
            crop_num = len(d_scores)
            c_scores = []
            c_classes_string = []
            c_ids = []
            for i in range(crop_num):
              	c_id, c_score, c_class_string = classifier.run_inference_on_image(d_images[i])
              	c_scores.append(c_score)
              	c_classes_string.append(c_class_string)
              	c_ids.append(c_id)
            vis_util.my_visualize_boxes_and_labels_on_image_array(
                image_np,
                d_boxes,
                c_scores,
                c_classes_string,
                c_ids,
                use_normalized_coordinates=True,
                line_thickness=6)
            plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)
