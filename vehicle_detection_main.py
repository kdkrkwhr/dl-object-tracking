import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
import sys
import copy
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import timeit

total_passed_vehicle = 0
MODEL_FILE = 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'

mouse_flag = True
mouse_x = -300
mouse_y = -300
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 5

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def mousecall(event, x, y, flags, param):
    global mouse_x, mouse_y,  first_frame, mouse_flag

    if event == cv2.EVENT_LBUTTONDOWN:  # Horizontal Line
        mouse_x = x
        mouse_y = y
        if mouse_flag == True:
            img = copy.deepcopy(first_frame)

            cv2.imshow('result', img)
        print('x - : ', x)
        print('y l : ', y)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Vertical Line
        if mouse_flag == True:
            img = copy.deepcopy(first_frame)
            cv2.imshow('result', img)

# Detection
def object_detection_function():
    global mouse_x, mouse_y, mouse_flag
    total_passed_vehicle = 0
    mouse_flag = False
    video = 'video/qq1.jpg'
    rt_x = 752
    lt_x = 703

    lb_x = 783
    rb_x = 891

    ori_car_width = 1.8
    cctv_h = 10          # CCTV 카메라 설치 높이
    lane_interval = 3    # 차선 간격
    dis_line_a = 50     # CCTV 카메라 부터 가까운 객체의 직선 거리

    pixel = ((rb_x - lb_x) * dis_line_a) / lane_interval   # 픽셀 값
    print('pixel :', pixel)
    dis_line_b = (pixel * lane_interval) / (rt_x - lt_x)      # CCTV 카메라 부터 먼 객체 사이의 직선 거리

    dis_a = ((dis_line_a ** 2) - (cctv_h ** 2)) ** 0.5  # CCTV 카메라 부터 가까운 객체 사이의 2차원 거리

    dis_b = ((dis_line_b ** 2) - (cctv_h ** 2)) ** 0.5  # CCTV 카메라 부터 먼 객체 사이의 2차원 거리

    distance = dis_b - dis_a

    print('가까운 영역 직선 거리 : ', dis_line_a)
    print('먼 영역 직선 거리 : ', dis_line_b)

    print('가까운 영역 거리 : ', dis_a)
    print('먼 영역 거리 : ', dis_b)

    print('최종 거리 : ', distance)
    print('대기 차량 대(수) : ', int(distance / 5))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            cap = cv2.VideoCapture(video)

            fps_flag = True
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps > 10:
                fps_flag = True
            elif fps < 10:
                fps_flag = False

            while cap.isOpened():
                cv2.setMouseCallback("result", mousecall)
                cv2.setMouseCallback("test1", mousecall)
                (ret, frame) = cap.read()
                if ret == True:

                    # if fps_flag == True:
                    #     if cap.get(1) % 5 != 0:
                    #         continue

                    input_frame = frame
                    image_np_expanded = np.expand_dims(input_frame, axis=0)

                    (boxes, scores, classes, num) = \
                        sess.run([detection_boxes, detection_scores,
                                 detection_classes, num_detections],
                                 feed_dict={image_tensor: image_np_expanded})

                    (counter, csv_line, road_num, counter_per, area_count, car_width, person_count) = \
                        vis_util.visualize_boxes_and_labels_on_image_array(
                        mouse_y,
                        cap.get(1),
                        input_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True, line_thickness=4)

                    # print('personCount : ', person_count)

                    pts = np.array([[lt_x, 450], [rt_x, 440], [rb_x, 695], [lb_x, 715]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(input_frame, [pts], True, (255, 255, 0), 2)

                    total_passed_vehicle = total_passed_vehicle + counter

                    if car_width != (0, 0):
                        print('car : ', car_width)
                        start_car_width = (pixel * ori_car_width) / round(car_width[0])   # Near 객체 폭
                        end_car_width = (pixel * ori_car_width) / round(car_width[1])     # Far 객체 폭

                        start_car_width = ((start_car_width ** 2) - (cctv_h ** 2)) ** 0.5
                        end_car_width = ((end_car_width ** 2) - (cctv_h ** 2)) ** 0.5

                        # print('먼 객체 거리 : ', end_car_width)
                        # print('가까운 객체 거리 : ', start_car_width)

                        print('차선 사이 최종 거리) : ', end_car_width - start_car_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.imshow('result', input_frame)

                    # 투시 변환 (매개 좌표 4)
                    perspective_pt_a = np.float32([[lt_x, 450], [lb_x, 715], [rt_x, 440], [rb_x, 695]])
                    perspective_pt_b = np.float32([[0, 0], [0, int(distance * 20)], [lane_interval * 20, 0], [lane_interval * 20, int(distance * 20)]])

                    M = cv2.getPerspectiveTransform(perspective_pt_a, perspective_pt_b)

                    test1 = cv2.warpPerspective(input_frame, M, (lane_interval * 20, int(distance * 20)))

                    test1 = cv2.transpose(test1)  # 행렬 변경
                    test1 = cv2.flip(test1, 1)

                    cv2.imshow('test1', test1)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        break

                    cv2.waitKey(0)

                    if csv_line != 'not_available':
                        with open('traffic_measurement.csv', 'a') as f:
                            writer = csv.writer(f)
                            (size, color, direction, speed) = csv_line.split(',')
                            writer.writerows([csv_line.split(',')])
                else:
                    # print("- END -")
                    cap = cv2.VideoCapture(video)
                    continue

object_detection_function()