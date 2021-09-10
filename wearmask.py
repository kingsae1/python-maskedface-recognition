import os
import sys
import argparse
import numpy as np
import cv2
import re

import math
import dlib
from PIL import Image, ImageFile

__version__ = '0.3.0'

# Define Constant
IMAGE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'mask_images')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')

error_count = 0
success_count = 0

# Create Rectangle


def rect_to_bbox(rect):
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


def print_result(success, error):
    global success_count
    global error_count

    if success:
        success_count += 1
    elif error:
        error_count += 1


def face_alignment(faces):
    # 예측하기
    predictor = dlib.shape_predictor(
        "D:\Workspaces\python.recognition.face\wear_mask_to_face\models\shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 두 눈의 중점을 구하기
        eye_center = ((shape.part(36).x + shape.part(45).x) *
                      1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 각도
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 연산
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 회전 행렬
        RotImg = cv2.warpAffine(
            face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned


def cli(pic_path='', save_pic_path='', model_type='hog'):
    parser = argparse.ArgumentParser(
        description='Wear a face mask in the given picture.')
    # parser.add_argument('pic_path', default='/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',help='Picture path.')
    # parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')

    # 모델 HOG(Histogram of Oriented Gradients), 학습된 CNN모델
    # HOG는 픽셀 값의 변화로 영상 밝기의 변화의 방향을 그라디언트로 표현 하여 객체 탐색

    parser.add_argument('--model', default=model_type,
                        choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)

    color = 'defult'

    if args.black:
        mask_path = BLACK_IMAGE_PATH
        color = 'black'
    elif args.blue:
        mask_path = BLUE_IMAGE_PATH
        color = 'blue'
    elif args.red:
        mask_path = RED_IMAGE_PATH
        color = 'red'
    else:
        mask_path = DEFAULT_IMAGE_PATH
        color = 'default'

    # mask_path = BLUE_IMAGE_PATH
    save_pic_path = re.sub('.png|.jpg', '_' + color+'.png', save_pic_path)
    # print('PATH : ' + save_pic_path)
    FaceMasker(pic_path, mask_path, True, model_type, save_pic_path).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog', save_path='/results'):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(
            face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(
            face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            for (i, rect) in enumerate(face_locations):
                src_face_num = src_face_num + 1
                (x, y, w, h) = rect_to_bbox(rect)
                detect_face = with_mask_face[y:y + h, x:x + w]
                src_faces.append(detect_face)

            faces_aligned = face_alignment(src_faces)
            face_num = 0
            for faces in faces_aligned:
                face_num = face_num + 1
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                size = (int(128), int(128))
                faces_after_resize = cv2.resize(
                    faces, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(self.save_path, faces_after_resize)
            # if self.show:
            #     self._face_img.show()
            # save
            # self._save()
            print_result(True, False)
        else:
            print_result(False, True)

        print("[Result] Success : " + str(success_count) +
              " , Error : " + str(error_count) + ' Path : ' + self.save_path)

    def _mask_face(self, face_landmark: dict):
        try:
            nose_bridge = face_landmark['nose_bridge']
            nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
            nose_v = np.array(nose_point)

            chin = face_landmark['chin']
            chin_len = len(chin)
            chin_bottom_point = chin[chin_len // 2]
            chin_bottom_v = np.array(chin_bottom_point)
            chin_left_point = chin[chin_len // 8]
            chin_right_point = chin[chin_len * 7 // 8]

            # split mask and resize
            width = self._mask_img.width
            height = self._mask_img.height
            width_ratio = 1.2
            new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

            # left
            mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
            mask_left_width = self.get_distance_from_point_to_line(
                chin_left_point, nose_point, chin_bottom_point)
            mask_left_width = int(mask_left_width * width_ratio)
            mask_left_img = mask_left_img.resize((mask_left_width, new_height))

            # right
            mask_right_img = self._mask_img.crop(
                (width // 2, 0, width, height))
            mask_right_width = self.get_distance_from_point_to_line(
                chin_right_point, nose_point, chin_bottom_point)
            mask_right_width = int(mask_right_width * width_ratio)
            mask_right_img = mask_right_img.resize(
                (mask_right_width, new_height))

            # merge mask
            size = (mask_left_img.width + mask_right_img.width, new_height)
            mask_img = Image.new('RGBA', size)
            mask_img.paste(mask_left_img, (0, 0), mask_left_img)
            mask_img.paste(
                mask_right_img, (mask_left_img.width, 0), mask_right_img)

            # rotate mask
            angle = np.arctan2(
                chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
            rotated_mask_img = mask_img.rotate(angle, expand=True)

            # calculate mask location
            center_x = (nose_point[0] + chin_bottom_point[0]) // 2
            center_y = (nose_point[1] + chin_bottom_point[1]) // 2

            offset = mask_img.width // 2 - mask_left_img.width
            radian = angle * np.pi / 180
            box_x = center_x + int(offset * np.cos(radian)) - \
                rotated_mask_img.width // 2
            box_y = center_y + int(offset * np.sin(radian)) - \
                rotated_mask_img.height // 2

            # add mask
            self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        except TypeError as e:
            print("[Error] " + str(e))
        except ValueError as e:
            print("[Error] " + str(e))
        except:
            print('[Error] No Found !!')

    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
            np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                    (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':

    # dataset_path = 'sample_test\sample_data'
    # save_dataset_path = "sample_test\sample_result_v3"
    dataset_path = "kaggle_data\lfw-deepfunneled"
    save_dataset_path = "kaggle_data\lfw-deepfunneled-result"

    colorArray = ['default', 'black', 'blue', 'red']

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            imgpath = os.path.join(root, name)
            new_root = os.getcwd() + '\\' + root
            new_root = new_root.replace(
                'lfw-deepfunneled', 'lfw-deepfunneled-result')
            save_image_path = os.path.join(new_root, name)

            if not os.path.exists(new_root):
                os.mkdir(new_root)

            isExist = False

            for color in colorArray:
                savedImagePath = re.sub(
                    '.png|.jpg', '_' + color+'.png', save_image_path)

                if not os.path.isfile(savedImagePath):
                    isExist = False
                else:
                    isExist = True
                    break

            # print('[Not Exist] ' + str(isExist) + ' File : ' + save_image_path)

            if isExist == False:
                cli(imgpath, save_image_path, 'hog')
