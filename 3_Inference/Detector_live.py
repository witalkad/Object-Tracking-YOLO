import os
import sys
import cv2
from openni import openni2
from openni import _openni2 as c_api

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

def get_pos(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global ix, jy
        (ix,jy)=(x,y)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        global openwin
        openwin=False

    return
src_path = os.path.join(get_parent_dir(1),'2_Training','src')
utils_path = os.path.join(get_parent_dir(1),'Utils')

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1),'Data')

image_folder = os.path.join(data_folder,'Source_Images')

image_test_folder = os.path.join(image_folder,'Test_Images')

detection_results_folder = os.path.join(image_folder,'Test_Image_Detection_Results') 
detection_results_file = os.path.join(detection_results_folder, 'Detection_Results.csv')

model_folder =  os.path.join(data_folder,'Model_Weights')

model_weights = os.path.join(model_folder,'trained_weights_final.h5')
model_classes = os.path.join(model_folder,'data_classes.txt')

anchors_path = os.path.join(src_path,'keras_yolo3','model_data','yolo_anchors.txt')

FLAGS = None

frame_cache=np.zeros((1,240,320))
temperature=np.array([0])
path = 'C:/Users/ELLAB_TEST/Python Projects/orbecc viewer/Orbecmini_321'

openni2.initialize("C:\Program Files\OpenNI-Windows-x64-2.3\Redist")
dev = openni2.Device.open_any()
depth_stream = dev.create_color_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_YUYV, resolutionX = 640, resolutionY = 480, fps = 30))
cx=0
cy=0

if __name__ == '__main__':
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        "--input_path", type=str, default=image_test_folder,
        help = "Path to image/video directory. All subdirectories will be included. Default is " + image_test_folder
    )

    parser.add_argument(
        "--output", type=str, default=detection_results_folder,
        help = "Output path for detection results. Default is " + detection_results_folder
    )

    parser.add_argument(
        "--no_save_img", default=False, action="store_true",
        help = "Only save bounding box coordinates but do not save output images with annotated boxes. Default is False."
    )

    parser.add_argument(
        "--file_types", '--names-list', nargs='*', default=[], 
        help = "Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4"
    )

    parser.add_argument(
        '--yolo_model', type=str, dest='model_path', default = model_weights,
        help='Path to pre-trained weight files. Default is ' + model_weights
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path', default = anchors_path,
        help='Path to YOLO anchors. Default is '+ anchors_path
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path', default = model_classes,
        help='Path to YOLO class specifications. Default is ' + model_classes
    )

    parser.add_argument(
        '--gpu_num', type=int, default = 1,
        help='Number of GPU to use. Default is 1'
    )

    parser.add_argument(
        '--confidence', type=float, dest = 'score', default = 0.25,
        help='Threshold for YOLO object confidence score to show predictions. Default is 0.25.'
    )


    parser.add_argument(
        '--box_file', type=str, dest = 'box', default = detection_results_file,
        help='File to save bounding box results to. Default is ' + detection_results_file
    )
    
    parser.add_argument(
        '--postfix', type=str, dest = 'postfix', default = '_catface',
        help='Specify the postfix for images with bounding boxes. Default is "_catface"'
    )
    

    FLAGS = parser.parse_args()

    save_img = not FLAGS.no_save_img

    file_types = FLAGS.file_types



    print('compiling model...')
    # define YOLO detector
    yolo = YOLO(**{"model_path": FLAGS.model_path,
                "anchors_path": FLAGS.anchors_path,
                "classes_path": FLAGS.classes_path,
                "score" : FLAGS.score,
                "gpu_num" : FLAGS.gpu_num,
                "model_image_size" : (416, 416),
                }
               )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, 'r')
    input_labels = [line.rstrip('\n') for line in class_file.readlines()]
    print('Found {} input labels: {} ...'.format(len(input_labels), input_labels))

    openwin = True
    while openwin:
        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_triplet()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (1, 480, 640)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        m = img.copy()
        n = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
        n = n.astype(np.uint8)
        print(n.shape)
        prediction, new_image = yolo.detect_image(n)
        cv2.imshow('image', new_image)
        cv2.setMouseCallback('image', get_pos)

        if cv2.waitKey(34) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    openni2.unload()
    # Close the current yolo session
    yolo.close_session()



