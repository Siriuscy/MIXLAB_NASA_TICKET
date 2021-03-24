from utils_.utils import body_img_path_map, paste, img_pipline, circle, combine_image_to_video, transform_video_to_image
import paddlehub as hub
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os

pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")


def pipline(pose_img_path):
    result = pose_estimation.keypoint_detection(paths=[pose_img_path])
    img_flag = Image.open(body_img_path_map['background']).convert('RGBA')
    proportion_shoulder = (1050 - 60) / (result[0]['data']['left_shoulder'][0] - result[0]['data']['right_shoulder'][0])

    #     read
    # body_img = Image.open(body_img_path_map['body']).convert('RGBA')
    # body_img = body_img.resize(
    #     (int(body_img.size[0] / proportion_shoulder), int(body_img.size[1] / proportion_shoulder)))
    # helmet_rear_img = Image.open(body_img_path_map['helmet_rear']).convert('RGBA')
    # helmet_rear_img = helmet_rear_img.resize(
    #     (int(helmet_rear_img.size[0] / proportion_shoulder), int(helmet_rear_img.size[1] / proportion_shoulder)))
    # helmet_front_img = Image.open(body_img_path_map['helmet_front']).convert('RGBA')
    # helmet_front_img = helmet_front_img.resize(
    #     (int(helmet_front_img.size[0] / proportion_shoulder), int(helmet_front_img.size[1] / proportion_shoulder)))
    # bag = Image.open(body_img_path_map['bag']).convert('RGBA')
    # bag = bag.resize(
    #     (int(bag.size[0] / proportion_shoulder), int(bag.size[1] / proportion_shoulder)))
    # face_img = circle(body_img_path_map['face'], int(helmet_front_img.size[1]))

    #     location
    # body_x = int(
    #     (result[0]['data']['left_shoulder'][0] + result[0]['data']['ç'][0]) / 2 - (body_img.size[0] / 2))
    # body_y = int(result[0]['data']['left_shoulder'][1] - 550 / 1144 * body_img.size[1])
    #
    # helmet_x = int((result[0]['data']['left_shoulder'][0] + result[0]['data']['right_shoulder'][0]) / 2 - (
    #         helmet_front_img.size[0] / 2))
    # helmet_y = int(result[0]['data']['left_shoulder'][1] - 1.2 * helmet_front_img.size[1])
    #
    # face_x = int((result[0]['data']['left_shoulder'][0] + result[0]['data']['right_shoulder'][0]) / 2 - (
    #         face_img.size[0] / 2))
    # face_y = int(result[0]['data']['left_shoulder'][1] - 1.23 * face_img.size[1])
    #

    # test
    # head
    img_flag = img_pipline(body_img_path_map['left_elbow'], img_flag, result[0]['data']['left_shoulder'],
                           result[0]['data']['left_elbow'], key_y_proportion=1 / 4,
                           img_proportion=proportion_shoulder)
    img_flag = img_pipline(body_img_path_map['right_elbow'], img_flag, result[0]['data']['right_shoulder'],
                           result[0]['data']['right_elbow'], key_y_proportion=1 / 3,
                           img_proportion=proportion_shoulder)
    img_flag = img_pipline(body_img_path_map['body'], img_flag, result[0]['data']['upper_neck'],
                           result[0]['data']['pelvis'], key_y_proportion=1 / 3,
                           img_proportion=proportion_shoulder)
    img_flag = img_pipline(body_img_path_map['helmet_rear'], img_flag, result[0]['data']['head_top'],
                           result[0]['data']['upper_neck'], key_y_proportion=1 / 2,
                           img_proportion=proportion_shoulder)

    img_flag = img_pipline(body_img_path_map['face'], img_flag, result[0]['data']['head_top'],
                           result[0]['data']['upper_neck'], key_y_proportion=12 / 20,
                           img_proportion=proportion_shoulder)
    img_flag = img_pipline(body_img_path_map['helmet_front'], img_flag, result[0]['data']['head_top'],
                           result[0]['data']['upper_neck'], key_y_proportion=1 / 2,
                           img_proportion=proportion_shoulder)

    # img_flag = paste(bag, img_flag, body_x-13, body_y-30)

    # img_flag = paste(body_img, img_flag, body_x, body_y)
    # img_flag = paste(helmet_rear_img, img_flag, helmet_x, helmet_y)
    # img_flag = paste(face_img, img_flag, face_x, face_y)
    # img_flag = paste(helmet_front_img, img_flag, helmet_x, helmet_y)

    img_flag = img_pipline(body_img_path_map['left_wrist'], img_flag, result[0]['data']['left_elbow'],
                           result[0]['data']['left_wrist'], key_y_proportion=150 / 1139,
                           img_proportion=proportion_shoulder)
    img_flag = img_pipline(body_img_path_map['right_wrist'], img_flag, result[0]['data']['right_elbow'],
                           result[0]['data']['right_wrist'], key_y_proportion=150 / 1139,
                           img_proportion=proportion_shoulder)

    ticket = Image.open(body_img_path_map['ticket']).convert("RGBA")
    img_flag = paste(ticket, img_flag, 0, 0)
    return img_flag


def analysis_pose(input_frame_path, output_frame_path, is_print=True):
    '''
    分析图片中的人体姿势， 并转换为皮影姿势，输出结果
    '''
    file_items = os.listdir(input_frame_path)
    file_len = len(file_items)
    file_items.sort()
    for i in range(file_len):
        if is_print:
            print(i + 1, '/', file_len, ' ', os.path.join(output_frame_path, str(i) + '.jpg'))
        try:
            combine_img = pipline(os.path.join(input_frame_path, str(i) + '.jpg'))
            combine_img.save(os.path.join(output_frame_path, str(i) + '.png'))
        except:
            continue


if __name__ == '__main__':
    # 第一步：把视频切帧
    # fps = transform_video_to_image('./mp4/demo_1.mp4', './mp4_img/')
    # 第二步：每一帧都生成一张船票
    # analysis_pose('./mp4_img', './out_put', is_print=True)
    # 第三步：做成视频:
    # combine_image_to_video('./out_put', './out.mp4', 30)

    # output one pic
    img = pipline('./mp4_img/17.jpg')
    print(img.size)
    img.save('./one_ticket/one.png')
    img.show()
