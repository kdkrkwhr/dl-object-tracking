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

total_passed_vehicle = 0
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'

mouse_flag = True
mouse_x = -300
mouse_y = -300


DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
cap = cv2.VideoCapture('TCV_DJI_0006_2.mp4')
(ret, first_frame) = cap.read()

# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map,
#         max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         print('1')
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(392, 113)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#         self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
#         self.gridLayout.setObjectName("gridLayout")
#         self.verticalLayout = QtWidgets.QVBoxLayout()
#         self.verticalLayout.setObjectName("verticalLayout")
#         self.pushButton = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton.setObjectName("pushButton")
#         self.pushButton.clicked.connect(object_detection_function)
#
#         self.verticalLayout.addWidget(self.pushButton)
#         self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_2.setObjectName("pushButton_2")
#         self.verticalLayout.addWidget(self.pushButton_2)
#         self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.menubar = QtWidgets.QMenuBar(MainWindow)
#         self.menubar.setGeometry(QtCore.QRect(0, 0, 392, 21))
#         self.menubar.setObjectName("menubar")
#         MainWindow.setMenuBar(self.menubar)
#         self.statusbar = QtWidgets.QStatusBar(MainWindow)
#         self.statusbar.setObjectName("statusbar")
#         MainWindow.setStatusBar(self.statusbar)
#
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#
#     def retranslateUi(self, MainWindow):
#         print('2')
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
#         self.pushButton.setText(_translate("MainWindow", "START"))
#         self.pushButton_2.setText(_translate("MainWindow", "STOP"))
#
#
# def mousecall(event, x, y, flags, param):
#     print('3')
#     global mouse_x, mouse_y,  first_frame, mouse_flag
#
#     if event == cv2.EVENT_LBUTTONDOWN:  # Horizontal Line
#         mouse_x = x
#         mouse_y = y
#         if mouse_flag == True:
#             img = copy.deepcopy(first_frame)
#             cv2.line(img, (0, y), (1400, y), (0, 0xFF, 0), 5)
#             cv2.line(img, (x, y-20), (x, y+20), (0, 0xFF, 0), 5)
#             cv2.imshow('result', img)
#         print('x : ', x)
#         print('y : ', y)
#
#     elif event == cv2.EVENT_RBUTTONDOWN:  # Vertical Line
#         if mouse_flag == True:
#             img = copy.deepcopy(first_frame)
#             cv2.imshow('result', img)
#
# # Detection
# def object_detection_function(self):
#     global mouse_x, mouse_y, mouse_flag
#     print('4')
#     total_passed_vehicle = 0
#     road_num1 = 0
#     road_num2 = 0
#     road_num3 = 0
#     road_num4 = 0
#     road_num5 = 0
#     road_num6 = 0
#     road_num7 = 0
#     road_num8 = 0
#     speed = 'waiting...'
#     direction = 'waiting...'
#     size = 'waiting...'
#     color = 'waiting...'
#
#     mouse_flag = False
#
#     with detection_graph.as_default():
#         with tf.Session(graph=detection_graph) as sess:
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#
#             cap = cv2.VideoCapture('TCV_DJI_0006_2.avi')
#             fps_flag = True
#             fps = cap.get(cv2.CAP_PROP_FPS)
#
#             if fps > 10:
#                 fps_flag = True
#             elif fps < 10:
#                 fps_flag = False
#
#             while cap.isOpened():
#                 cv2.setMouseCallback("result", mousecall)
#                 (ret, frame) = cap.read()
#
#                 # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
#
#                 if ret == True:
#                     if fps_flag == True:
#                         if cap.get(1) % 5 != 0:
#                             continue
#                     input_frame = frame
#                     # crop = input_frame[bottomPlus:550, 400:520]
#
#                     image_np_expanded = np.expand_dims(input_frame, axis=0)
#
#                     (boxes, scores, classes, num) = \
#                         sess.run([detection_boxes, detection_scores,
#                                  detection_classes, num_detections],
#                                  feed_dict={image_tensor: image_np_expanded})
#
#                     (counter, csv_line, road_num) = \
#                         vis_util.visualize_boxes_and_labels_on_image_array(
#                         mouse_y,
#                         cap.get(1),
#                         input_frame,
#                         np.squeeze(boxes),
#                         np.squeeze(classes).astype(np.int32),
#                         np.squeeze(scores),
#                         category_index,
#                         use_normalized_coordinates=True, line_thickness=4)
#
#                     if road_num == "1":
#                         road_num1 = road_num1 + counter
#                     elif road_num == "2":
#                         road_num2 = road_num2 + counter
#                     elif road_num == "3":
#                         road_num3 = road_num3 + counter
#                     elif road_num == "4":
#                         road_num4 = road_num4 + counter
#                     elif road_num == "5":
#                         road_num5 = road_num5 + counter
#                     elif road_num == "6":
#                         road_num6 = road_num6 + counter
#                     elif road_num == "7":
#                         road_num7 = road_num7 + counter
#                     elif road_num == "8":
#                         road_num8 = road_num8 + counter
#
#                     total_passed_vehicle = total_passed_vehicle + counter
#
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     cv2.putText(input_frame, 'TOTAL Counting : ' + str(total_passed_vehicle), (10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '1st Counting : ' + str(road_num1), (10, 85), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '2st Counting : ' + str(road_num2), (10, 115), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '3st Counting : ' + str(road_num3), (10, 145), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '4st Counting : ' + str(road_num4), (10, 175), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '5st Counting : ' + str(road_num5), (10, 215), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '6st Counting : ' + str(road_num6), (10, 245), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '7st Counting : ' + str(road_num7), (10, 275), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#                     cv2.putText(input_frame, '8st Counting : ' + str(road_num8), (10, 305), font, 0.6, (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#
#                     cv2.putText(input_frame, 'Lane Move : ' + "0", (1100, 85), font, 0.6, (0, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
#
#                     if counter == 1:
#                         cv2.line(input_frame, (0, mouse_y), (1400, mouse_y), (0, 0xFF, 0), 5)
#                     else:
#                         cv2.line(input_frame, (0, mouse_y), (1400, mouse_y), (0, 0, 0xFF), 5)
#
#                     cv2.line(input_frame, (mouse_x, mouse_y-20), (mouse_x, mouse_y+20), (0, 0xFF, 0), 5)
#
#                     cv2.line(input_frame, (310, mouse_y-20), (310, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (390, mouse_y-20), (390, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (460, mouse_y-20), (460, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (530, mouse_y-20), (530, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (600, mouse_y-20), (600, mouse_y+20), (0, 0xFF, 0), 5)
#
#                     cv2.line(input_frame, (650, mouse_y-20), (650, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (720, mouse_y-20), (720, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (800, mouse_y-20), (800, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (870, mouse_y-20), (870, mouse_y+20), (0, 0xFF, 0), 5)
#                     cv2.line(input_frame, (950, mouse_y-20), (950, mouse_y+20), (0, 0xFF, 0), 5)
#
#                     # cv2.rectangle(input_frame, (10, 325), (230, 387), (180, 132, 109), -1)
#                     # cv2.putText(input_frame, 'ROI Line', (100, 570), font, 0.8, (0, 0, 0xFF), 2, cv2.LINE_AA)
#                     # cv2.putText(input_frame, 'LAST PASSED VEHICLE INFO', (11, 340), font, 0.5, (0xFF, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
#                     # cv2.putText(input_frame, '-Movement Direction: ' + direction, (14, 352), font, 0.4, (0xFF, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
#                     # cv2.putText(input_frame, '-Speed(km/h): ' + speed, (14, 362), font, 0.4, (0xFF, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
#                     # cv2.putText(input_frame, '-Color: ' + color, (14, 372), font, 0.4, (0xFF, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
#                     # cv2.putText(input_frame, '-Vehicle Size/Type: ' + size, (14, 382), font, 0.4, (0xFF, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
#
#                     cv2.imshow('result', input_frame)
#
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         cap.release()
#                         break
#
#                     if csv_line != 'not_available':
#                         with open('traffic_measurement.csv', 'a') as f:
#                             writer = csv.writer(f)
#                             (size, color, direction, speed) = csv_line.split(',')
#                             writer.writerows([csv_line.split(',')])
#
#                 else :
#                     print("- END -")
#                     cap = cv2.VideoCapture('TCV_DJI_0006_2.avi')
#                     continue

# object_detection_function()

# if __name__ == "__main__":
#
#     cv2.imshow('result', first_frame)
#     cv2.setMouseCallback("result", mousecall)
#
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
#     ------------------------------------------------------------------------------------------------------------------

class ShowVideo(QtCore.QObject):
    global mouse_y
    camera = cv2.VideoCapture('TCV_DJI_0006_2.mp4')

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        total_passed_vehicle = 0
        road_num1 = 0
        road_num2 = 0
        road_num3 = 0
        road_num4 = 0
        road_num5 = 0
        road_num6 = 0
        road_num7 = 0
        road_num8 = 0
        speed = 'waiting...'
        direction = 'waiting...'
        size = 'waiting...'
        color = 'waiting...'
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

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                camera = cv2.VideoCapture('TCV_DJI_0006_2.mp4')

                while camera.isOpened():
                    ret, image = camera.read()

                    image = cv2.resize(image, (800, 400), interpolation=cv2.INTER_AREA)

                    height, width = image.shape[:2]

                    if ret == True:
                        input_frame = image

                        image_np_expanded = np.expand_dims(input_frame, axis=0)

                        (boxes, scores, classes, num) = \
                            sess.run([detection_boxes, detection_scores,
                                      detection_classes, num_detections],
                                     feed_dict={image_tensor: image_np_expanded})

                        (counter, csv_line, road_num) = \
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                500,
                                camera.get(1),
                                input_frame,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True, line_thickness=4)

                        if road_num == "1":
                            road_num1 = road_num1 + counter
                        elif road_num == "2":
                            road_num2 = road_num2 + counter
                        elif road_num == "3":
                            road_num3 = road_num3 + counter
                        elif road_num == "4":
                            road_num4 = road_num4 + counter
                        elif road_num == "5":
                            road_num5 = road_num5 + counter
                        elif road_num == "6":
                            road_num6 = road_num6 + counter
                        elif road_num == "7":
                            road_num7 = road_num7 + counter
                        elif road_num == "8":
                            road_num8 = road_num8 + counter

                        total_passed_vehicle = total_passed_vehicle + counter

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(input_frame, 'TOTAL Counting : ' + str(total_passed_vehicle), (10, 35), font, 0.8,
                                    (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '1st Counting : ' + str(road_num1), (10, 85), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '2st Counting : ' + str(road_num2), (10, 115), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '3st Counting : ' + str(road_num3), (10, 145), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '4st Counting : ' + str(road_num4), (10, 175), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '5st Counting : ' + str(road_num5), (10, 215), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '6st Counting : ' + str(road_num6), (10, 245), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '7st Counting : ' + str(road_num7), (10, 275), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame, '8st Counting : ' + str(road_num8), (10, 305), font, 0.6,
                                    (0xFF, 0, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)

                        if counter != 1:
                            cv2.line(input_frame, (0, 400), (1400, 400), (0, 0xFF, 0), 5)
                            cv2.line(input_frame, (0, 500), (1400, 500), (0, 0xFF, 0), 5)
                        else:
                            cv2.line(input_frame, (0, 400), (1400, 400), (0, 0, 0xFF), 5)
                            cv2.line(input_frame, (0, 500), (1400, 500), (0, 0, 0xFF), 5)

                        color_swapped_image = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                        # input_frame = cv2.resize(input_frame, dsize=(1500, 800), interpolation=cv2.INTER_AREA)

                        qt_image = QtGui.QImage(color_swapped_image.data,
                                                width,
                                                height,
                                                color_swapped_image.strides[0],
                                                QtGui.QImage.Format_RGB888)

                        self.VideoSignal1.emit(qt_image)

                    else:
                        print("- END -")
                        camera = cv2.VideoCapture('TCV_DJI_0006_2.mp4')
                        continue


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def mousePressEvent(self, event):
        mouse_y = event.y
        print("eventY : ", event.y())

    def initUI(self):
        self.setWindowTitle('TCV')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):

        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)

    push_button1 = QtWidgets.QPushButton('Start')
    push_button1.clicked.connect(vid.startVideo)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setFixedWidth(800)
    main_window.setFixedHeight(450)
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())