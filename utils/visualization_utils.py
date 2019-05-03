#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""

# Imports
import collections
import functools
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf
import cv2
import numpy
import os
import timeit
from utils.image_utils import image_saver

from utils.speed_and_direction_prediction_module import speed_prediction

from utils.color_recognition_module import color_recognition_api

is_vehicle_detected = [0]
ROI_POSITION = 0

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

current_path = os.getcwd()
startCarWidth = 0
endCarWidth = 0
startLine = (0, 0)
endLine = (0, 0)
topSize = 0

areaCount = 0

# No 1
def visualize_boxes_and_labels_on_image_array(mouse_y,
                                              current_frame_number,  # 해당 Frame 번호
                                              image,  # 해당 Frame
                                              boxes,  # Object 좌표점 (4 개)
                                              classes,  # Object 번호
                                              scores,  # Object 가 맞을 확률
                                              category_index,  # 객체 카테고리 번호
                                              instance_masks=None,
                                              keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=100,
                                              min_score_thresh=0.4,
                                              agnostic_mode=False,
                                              line_thickness=4):

  global areaCount, startLine, endLine, topSize
  startLine = (0, 0)
  endLine = (0, 0)
  topSize = 0
  personCount = 0
  car_width = (0, 0)

  areaCount = 0
  csv_line_util = "not_available"
  counter = 0
  counter_per = 0
  is_vehicle_detected = []
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_keypoints_map = collections.defaultdict(list)

  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):  # 하나의 Frame 에서 Detecting 된 Object 를 ,표출할 Object 로 재분류
    if classes[i] == 1:
        personCount = personCount + 1

    if scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())

      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]

      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])

      if scores is None:
        box_to_color_map[box] = 'black'

      else:
        if not agnostic_mode:
          if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']
          else:
            class_name = 'NAME/A'
          display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
          display_str = '{}'.format(class_name)
        else:
          display_str = 'score: {}%'.format(int(100 * scores[i]))
        box_to_display_str_map[box].append(display_str)

        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]
  road_num = "0"
  print('classes[i] : ', classes[i])
  # 저장한 Object 좌표/컬러 를 담은 배열에 대한 처리
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box

    display_str_list = box_to_display_str_map[box]
    # 저장한 Object 좌표/컬러 를 담은 배열 중 car, truck, bus 객체만 표출 해주기 위한 분기 처리

    if (("car" in display_str_list[0]) or ("person" in display_str_list[0]) or ("bus" in display_str_list[0])):

            counter_per = counter_per + 1
            is_vehicle_detected, csv_line, update_csv, road_num, car_width = draw_bounding_box_on_image_array(
                mouse_y,
                current_frame_number,
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)

  if(1 in is_vehicle_detected):
    counter = 1
    del is_vehicle_detected[:]

    csv_line_util = class_name + "," + csv_line

  return counter, csv_line_util, road_num, counter_per, areaCount, car_width, personCount

# No 2
def draw_bounding_box_on_image_array(
                                     mouse_y,
                                     current_frame_number,
                                     image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    is_vehicle_detected, csv_line, update_csv, road_num, car_width = draw_bounding_box_on_image(mouse_y, current_frame_number, image_pil, ymin, xmin,
                                                                           ymax, xmax, color, thickness,
                                                                           display_str_list, use_normalized_coordinates)

    np.copyto(image, np.array(image_pil))
    return is_vehicle_detected, csv_line, update_csv, road_num, car_width

# No 3  객체에 분별한 마스크 덮는 함수
def draw_bounding_box_on_image(mouse_y,
                               current_frame_number,
                               image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):

    global areaCount, startLine, endLine, topSize, startCarWidth, endCarWidth
    im_width, im_height = image.size
    car_width = (0, 0)

    laneImg = np.zeros((im_height, im_width, 1), np.uint8)
    laneImg2 = np.zeros((im_height, im_width, 1), np.uint8)
    laneImg3 = np.zeros((im_height, im_width, 1), np.uint8)
    laneImg4 = np.zeros((im_height, im_width, 1), np.uint8)

    carImg = np.zeros((im_height, im_width, 1), np.uint8)

    lanePts = np.array([[703, 450], [771, 450], [891, 715], [783, 715]], np.int32)
    lanePts = lanePts.reshape((-1, 1, 2))
    laneArea = cv2.fillPoly(laneImg, [lanePts], (255, 255, 255))

    lanePts2 = np.array([[640, 450], [700, 450], [783, 715], [670, 715]], np.int32)
    lanePts2 = lanePts2.reshape((-1, 1, 2))
    laneArea2 = cv2.fillPoly(laneImg2, [lanePts2], (255, 255, 255))

    lanePts3 = np.array([[770, 450], [829, 450], [1000, 715], [894, 715]], np.int32)
    lanePts3 = lanePts3.reshape((-1, 1, 2))
    laneArea3 = cv2.fillPoly(laneImg3, [lanePts3], (255, 255, 255))

    lanePts4 = np.array([[832, 450], [890, 450], [1115, 715], [1003, 715]], np.int32)
    lanePts4 = lanePts4.reshape((-1, 1, 2))
    laneArea4 = cv2.fillPoly(laneImg4, [lanePts4], (255, 255, 255))

    carPts = np.array([[xmin * im_width, ymax * im_height], [xmax * im_width, ymax * im_height], [xmax * im_width, ymin * im_height], [xmin * im_width, ymin * im_height]], np.int32)
    carPts = carPts.reshape((-1, 1, 2))
    carArea = cv2.fillPoly(carImg, [carPts], (255, 255, 255))

    carSize = cv2.countNonZero(carArea) / 2  # 지정영역안에 객체가 50% 만 들어오게되도 지정영역안으로 간주

    result = cv2.bitwise_and(laneArea, carArea)
    points_result = cv2.countNonZero(result)

    result2 = cv2.bitwise_and(laneArea2, carArea)
    points_result2 = cv2.countNonZero(result2)

    result3 = cv2.bitwise_and(laneArea3, carArea)
    points_result3 = cv2.countNonZero(result3)

    result4 = cv2.bitwise_and(laneArea4, carArea)
    points_result4 = cv2.countNonZero(result4)

    update_csv = False
    is_vehicle_detected = [0]
    draw = ImageDraw.Draw(image)

    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if points_result > carSize:  # 차선영역내 객체 진입
        areaCount = areaCount + 1   # 차선영역내 객체 카운트

        if topSize < top:   # 들어온 객체 중 큰값 (가까운 객체)
            startLine = ((right + left) / 2, bottom)
            startCarWidth = right - left

        if areaCount <= 2:  # 들어온 객체가 두개 이상 일 경우
            if (startLine != (0, 0)) and (topSize > top):   # 우선 진입한 객체의 Y 좌표 값 이 높을 경우 (먼 객체)
                endLine = ((right + left) / 2, bottom)
                draw.line([startLine, endLine], width=thickness, fill=(0, 255, 0))  # 대기행렬 위한 Line
                endCarWidth = right - left
                car_width = (startCarWidth, endCarWidth)
                print('차 폭 (Near, Far) : ', car_width)

        topSize = top
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=(255, 255, 0))

    elif points_result2 > carSize:
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=(255, 255, 255))
    elif points_result3 > carSize:
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=(0, 255, 255))
    elif points_result4 > carSize:
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=(255, 0, 0))
    else:
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=(0, 0, 0))

    predicted_speed = "n.a."
    predicted_direction = "n.a."

    image_temp = numpy.array(image)
    detected_vehicle_image = image_temp[int(top):int(bottom), int(left):int(right)]
    road_number = "0"

    road_num = "0"
    ROI_POSITION = mouse_y - 100
    if (bottom > ROI_POSITION and road_number != "0"):  # 좌표에 따른 스피드 및 카운트
         predicted_direction, predicted_speed, is_vehicle_detected, update_csv, road_num = speed_prediction.predict_speed(road_number,
                                                                                                                top,
                                                                                                                bottom,
                                                                                                                right,
                                                                                                                left,
                                                                                                                current_frame_number,
                                                                                                                detected_vehicle_image,
                                                                                                                ROI_POSITION)

    predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)

    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()

    display_str_list[0] = display_str_list[0]
    csv_line = predicted_color + "," + str(predicted_direction) + "," + str(predicted_speed)  # csv line created
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    for display_str in display_str_list[::-1]:

        text_width, text_height = font.getsize(display_str)

        margin = np.ceil(0.05 * text_height)

        if points_result > carSize:
            draw.rectangle([
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom)],
                fill=(255, 255, 0))
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
        elif points_result2 > carSize:
            draw.rectangle([
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom)],
                fill=(255, 255, 255))
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
        elif points_result3 > carSize:
            draw.rectangle([
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom)],
                fill=(0, 255, 255))
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
        elif points_result4 > carSize:
            draw.rectangle([
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom)],
                fill=(255, 0, 0))
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
        else:
            draw.rectangle([
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom)],
                fill=(0, 0, 0))
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='white',
                font=font)
        text_bottom -= text_height - 2 * margin

        return is_vehicle_detected, csv_line, update_csv, road_num, car_width
