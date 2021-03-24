import os
import cv2
import paddle
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()


# os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True)

        assert os.path.exists(
            './models/photo2cartoon_weights.pdparams'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = paddle.load('./models/photo2cartoon_weights.pdparams')
        self.net.set_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face * mask + (1 - mask) * 255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = paddle.to_tensor(face)

        # inference
        with paddle.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon


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
            cv2.imwrite(img_path + '%d.jpg' % count, frame)
            count += 1
        else:
            break
    video_capture.release()
    print('视频图片保存成功, 共有 %d 张' % count)
    return fps


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
            img = cv2.cvtColor(cv2.imread(os.path.join(input_frame_path, str(1) + '.jpg')), cv2.COLOR_BGR2RGB)
            c2p = Photo2Cartoon()
            cartoon = c2p.inference(img)
            if cartoon is not None:
                cv2.imwrite(os.path.join(output_frame_path, str(i) + '.jpg'), cartoon)
        except:
            continue


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
            pic_name = os.path.join(comb_path, str(i) + ".jpg")
            if is_print:
                print(i + 1, '/', file_len, ' ', pic_name)
            img = cv2.imread(pic_name)
            out.write(img)
        out.release()


if __name__ == '__main__':
    # 第一步：把视频切帧
    # fps = transform_video_to_image('./face_mp4/face2.mp4', './face_img/')
    # analysis_pose('./face_img/', './face_img_gan/', is_print=True)
    combine_image_to_video('./face_img_gan/', './face.mp4', 30)
