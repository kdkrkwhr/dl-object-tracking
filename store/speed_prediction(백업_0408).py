#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
bottom_position_of_detected_vehicle = 0

bottom_position_of_detected_vehicle1 = [0]
bottom_position_of_detected_vehicle2 = [0]
bottom_position_of_detected_vehicle3 = [0]
bottom_position_of_detected_vehicle4 = [0]
bottom_position_of_detected_vehicle5 = [0]
bottom_position_of_detected_vehicle6 = [0]
bottom_position_of_detected_vehicle7 = [0]
bottom_position_of_detected_vehicle8 = [0]

def predict_speed(
    road_number,
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    ):

    speed = 'n.a.'
    direction = 'n.a.'
    scale_constant = 1
    isInROI = True
    update_csv = False
    if bottom < 450:
        scale_constant = 1

    elif bottom > 450:
        scale_constant = 2

    else:
        isInROI = False

    if road_number == "1":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle1[0]
    elif road_number == "2":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle2[0]
    elif road_number == "3":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle3[0]
    elif road_number == "4":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle4[0]
    elif road_number == "5":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle5[0]
    elif road_number == "6":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle6[0]
    elif road_number == "7":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle7[0]
    elif road_number == "8":
        bottom_position_of_detected_vehicle = bottom_position_of_detected_vehicle8[0]

    if (roi_position + 120) < bottom_position_of_detected_vehicle  and bottom_position_of_detected_vehicle < (roi_position + 140) and roi_position < bottom:
        is_vehicle_detected.insert(0, 1) # 카운트 플래그 1을 생성
        update_csv = True
        image_saver.save_image(crop_img)

    if bottom > bottom_position_of_detected_vehicle:
        direction = 'down'
    else:
        direction = 'up'

    if isInROI:
        if road_number == "1":
            pixel_length = bottom - bottom_position_of_detected_vehicle1[0]
        elif road_number == "2":
            pixel_length = bottom - bottom_position_of_detected_vehicle2[0]
        elif road_number == "3":
            pixel_length = bottom - bottom_position_of_detected_vehicle3[0]
        elif road_number == "4":
            pixel_length = bottom - bottom_position_of_detected_vehicle4[0]
        elif road_number == "5":
            pixel_length = bottom - bottom_position_of_detected_vehicle5[0]
        elif road_number == "6":
            pixel_length = bottom - bottom_position_of_detected_vehicle6[0]
        elif road_number == "7":
            pixel_length = bottom - bottom_position_of_detected_vehicle7[0]
        elif road_number == "8":
            pixel_length = bottom - bottom_position_of_detected_vehicle8[0]

        # scale_real_length = pixel_length * 44
        scale_real_length = pixel_length * 10000

        total_time_passed = current_frame_number - current_frame_number_list[0]
        # scale_real_time_passed = total_time_passed * 24
        scale_real_time_passed = total_time_passed * 60 * 60

        if scale_real_time_passed != 0:
            # speed = scale_real_length / scale_real_time_passed / scale_constant
            speed = scale_real_length / scale_real_time_passed

            # speed = speed / 6 * 40

            current_frame_number_list.insert(0, current_frame_number)

            if road_number == "1":
                bottom_position_of_detected_vehicle1.insert(0, bottom)
            elif road_number == "2":
                bottom_position_of_detected_vehicle2.insert(0, bottom)
            elif road_number == "3":
                bottom_position_of_detected_vehicle3.insert(0, bottom)
            elif road_number == "4":
                bottom_position_of_detected_vehicle4.insert(0, bottom)
            elif road_number == "5":
                bottom_position_of_detected_vehicle5.insert(0, bottom)
            elif road_number == "6":
                bottom_position_of_detected_vehicle6.insert(0, bottom)
            elif road_number == "7":
                bottom_position_of_detected_vehicle7.insert(0, bottom)
            elif road_number == "8":
                bottom_position_of_detected_vehicle8.insert(0, bottom)

    return direction, speed, is_vehicle_detected, update_csv, road_number