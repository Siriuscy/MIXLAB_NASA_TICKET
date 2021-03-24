import os
import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import paddlehub as hub
from PIL import Image

pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")


def transform_video_to_image(video_file_path, img_path):
    '''
    将视频中每一帧保存成图片
    '''
    video_capture = cv2.VideoCapture(video_file_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    count = 0
    while (True):
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.resize(frame, (1729, 1153))
            cv2.imwrite(img_path + '%d.jpg' % count, frame)
            count += 1
        else:
            break
    video_capture.release()
    print('视频图片保存成功, 共有 %d 张' % count)
    return fps


def combine_image_to_video(comb_path, output_file_path, fps=30, is_print=False):
    '''
        合并图像到视频
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    file_items = os.listdir(comb_path)
    file_len = len(file_items)
    # print(comb_path, file_items)
    if file_len > 0:
        temp_img = cv2.imread(os.path.join(comb_path, file_items[0]))
        img_height, img_width = temp_img.shape[0], temp_img.shape[1]

        out = cv2.VideoWriter(output_file_path, fourcc, fps, (img_width, img_height))

        for i in range(file_len):
            pic_name = os.path.join(comb_path, str(i) + ".png")
            if is_print:
                print(i + 1, '/', file_len, ' ', pic_name)
            img = cv2.imread(pic_name)
            out.write(img)
        out.release()


def get_true_angel(value):
    '''
    转转得到角度值
    '''
    return value / np.pi * 180


def get_angle(x1, y1, x2, y2):
    '''
    计算旋转角度
    '''
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    result_angele = 0
    if x1 == x2:
        if y1 > y2:
            result_angele = 180
    else:
        if y1 != y2:
            the_angle = int(get_true_angel(np.arctan(dx / dy)))
        if x1 < x2:
            if y1 > y2:
                result_angele = -(180 - the_angle)
            elif y1 < y2:
                result_angele = -the_angle
            elif y1 == y2:
                result_angele = -90
        elif x1 > x2:
            if y1 > y2:
                result_angele = 180 - the_angle
            elif y1 < y2:
                result_angele = the_angle
            elif y1 == y2:
                result_angele = 90

    if result_angele < 0:
        result_angele = 360 + result_angele
    return result_angele


def rotate_bound(image, angle, key_point_y):
    '''
    旋转图像，并取得关节点偏移量
    '''
    #     转成np
    image = np.asarray(image)
    # 获取图像的尺寸
    (h, w) = image.shape[:2]
    # 旋转中心
    (cx, cy) = (w / 2, h / 2)
    # 关键点必须在中心的y轴上
    (kx, ky) = cx, key_point_y
    d = abs(ky - cy)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 计算旋转后的相对位移
    move_x = nW / 2 + np.sin(angle / 180 * np.pi) * d
    move_y = nH / 2 - np.cos(angle / 180 * np.pi) * d

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    image = cv2.warpAffine(image, M, (nW, nH))
    #     转成image
    image = Image.fromarray(image)
    return image, int(move_x), int(move_y)


def get_distences(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


body_img_path_map = {
    # "right_hip" : "./work/shadow_play_material/body.jpg",
    # "right_knee" : "./work/shadow_play_material/head.jpg",
    # "left_hip" : "./work/shadow_play_material/head.jpg",
    # "left_knee" : "./work/shadow_play_material/body.jpg",
    "left_elbow": "./img/left_elbow.png",
    "left_wrist": "./img/left_wrist_takeoff.png",
    "right_elbow": "./img/right_elbow.png",
    "right_wrist": './img/right_wrist_takeoff.png',
    "head": "./img/helmet_front.png",
    "body": "./img/body.png",
    'helmet_rear': './img/helmet_rear.png',
    'helmet_front': './img/helmet_front.png',
    'face': './img/face_1.jpeg',
    'background': './img/ticket_bottom.png',
    'bag': './img/bag.png',
    'ticket': './img/ticket_up.png'
}


def img_pipline(img_path, img_flag, point_a, point_b, key_y_proportion, img_proportion):
    '''
    const:原图两个点之间间距
    img_proportion_width：宽度的比值，自己看着来
    key_y_proportion:离y的比值
    '''
    #     resize:
    if img_path == './img/face_1.jpeg':
        img = circle(img_path, 782 / 3)
    else:
        img = Image.open(img_path).convert("RGBA")
        img = img.resize((int(img.size[0] / img_proportion), int(img.size[1] / img_proportion)))

    key_y = int(key_y_proportion * img.size[1])
    angle = get_angle(point_a[0], point_a[1], point_b[0], point_b[1])
    img_rotated, x, y = rotate_bound(img, angle=angle, key_point_y=key_y)
    img_flag = paste(img_rotated, img_flag, point_a[0] - x, point_a[1] - y)
    return img_flag


def paste(img, img_flag, zero_x, zero_y):
    r, g, b, a = img.split()
    img_flag.paste(img, (zero_x, zero_y), mask=a)
    return img_flag


def circle(img_path, resize_num):
    ima = Image.open(img_path).convert("RGBA")
    w, h = ima.size
    ima = ima.crop((w / 20, 0, 19 * w / 20, h * 9 / 10)).resize((int(resize_num * 0.9), int(resize_num * 0.9)))

    #     白色转透明度
    W, H = ima.size
    for h in range(H):
        for w in range(W):
            if ima.getpixel((w, h))[0] > 250 and ima.getpixel((w, h))[1] > 250 and ima.getpixel((w, h))[2] > 250:
                ima.putpixel((w, h), (255, 255, 255, 0))
    #     剪裁成圆形
    size = ima.size
    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    r3 = int(r2 / 2)
    imb = Image.new('RGBA', (r3 * 2, r3 * 2), (255, 255, 255, 0))
    pima = ima.load()
    pimb = imb.load()
    r = float(r2 / 2)
    for i in range(r2):
        for j in range(r2):
            lx = abs(i - r)
            ly = abs(j - r)
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5
            if l < r3:
                pimb[i - (r - r3), j - (r - r3)] = pima[i, j]
    return imb


if __name__ == "__main__":
    pass
