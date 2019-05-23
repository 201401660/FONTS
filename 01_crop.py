# -*- coding: utf-8 -*-
# crop template

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance


# list division
def chunksList(List,size):
    for i in range(0, len(List), size):
        yield List[i : i + size]

# template size 280 X 200 mm
# space size    1.5 X 1.5 mm

def crop_image_uniform(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    imgList = sorted(os.listdir(src_dir))

    # unicode list matching template image
    idx = 0
    f = open('399-uniform.txt', 'r')
    charList = f.readlines()
    charList = [i.strip() for i in charList] # strip : delete '\n'

    # crop image
    for img in imgList:
        image = cv2.imread(os.path.join(src_dir,img))

        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)

        # binary
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # cv2.namedWindow('second', cv2.WINDOW_NORMAL)
        # cv2.imshow('second', thresh)
        # cv2.waitKey(0)

        # dilation
        kernel = np.ones((13, 13), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        # cv2.namedWindow('dilated', cv2.WINDOW_NORMAL)
        # cv2.imshow('dilated', img_dilation)
        # cv2.waitKey(0)

        # find contours
        # opencv > 4.0 ------------ ctrs,_ =
        # opencv < 4.0 ------------ _, ctrs, _ =
        ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cnt = 0
        rect = []

        for i, ctr in enumerate(ctrs):
            # Get bounding
            x, y, w, h = cv2.boundingRect(ctr)

            #print("H", h / float(H), "w", w / float(W))

            # Getting ROI
            #roi = image[y:y + h, x:x + w]

            # show ROI
            # cv2.imshow('segment no:'+str(i),roi)

            if 0.045 <= h / float(H) < 0.9 and 0.065 <= w / float(W) < 0.9:
                rect.append([x, y, w, h])
                #print('xywh:', [x, y, w, h])

                cnt = cnt + 1

        # [ x,y,w,h ] 중 y 값에 대해 정렬 => row 만큼 묶을 수 있게 됨
        rect = sorted(rect, key=lambda k: [k[1]])

        rect = list(chunksList(rect, 12))  # 12개(= 1 row )만큼씩 묶어줌

        for row in rect:
            #print(row)
            row = sorted(row, key=lambda k: [k[0]])  # row별로 column(=x 좌표)에 대해 정렬
            for rec in row:

                x, y, w, h = rec
                roi = thresh[y:y + h + 1, x:x + w + 1]  # 1 is margin

                # To make contours on purpose
                open_kernel = np.ones((15,15), np.uint8)
                dilated_image = cv2.dilate(roi, open_kernel, iterations=1)
                dilated_image = ~dilated_image
                contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours.sort(key=cv2.contourArea)

                # not save image not detected contour
                try:
                    _r = [contour for contour in contours][-2]
                    #_r = sorted([contour for contour in contours if cv2.contourArea(contour)<16000])[-1]

                except:
                    idx += 1
                    continue
                # for contour in contours:
                #     _x, _y, _w, _h= cv2.boundingRect(contour)
                #     img1 = cv2.rectangle(dilated_image, (_x, _y), (_x + _w, _y + _h), 10)
                # cv2.imshow('test', img1)
                # cv2.waitKey(0)
                _x, _y, _w, _h = cv2.boundingRect(_r)

                #img1 = cv2.rectangle(dilated_image, (_x, _y), (_x + _w, _y + _h), 7)


                # make empty white image
                background = np.zeros([128,128],dtype=np.uint8)
                background.fill(255)

                _roi = gray[y:y + h + 1, x:x + w + 1]

                # cv2.rectangle(image, (x, y), (x + w, y + h), (171, 255, 0), 3)
                # cv2.namedWindow('marked areas', cv2.WINDOW_NORMAL)
                # cv2.imshow('marked areas', image)
                # cv2.waitKey(0)


                cropped_image = _roi[_y:_y+_h, _x:_x+_w]


                # cv2.rectangle(_roi, (_x, _y), (_x + _w, _y + _h), (100, 100, 100), 1)
                # cv2.namedWindow('cropped areas', cv2.WINDOW_NORMAL)
                # cv2.imshow('cropped areas', _roi)
                # cv2.waitKey(0)


                # resize according to width and height
                if _w > _h:
                    height_size = 128 * _h / _w
                    #print('h',height_size)
                    cropped_image = cv2.resize(cropped_image,(128, height_size))
                    background[int(64 - height_size / 2):int(64 - height_size / 2) + cropped_image.shape[0],:cropped_image.shape[1]] = cropped_image
                else:
                    width_size = 128 * _w / _h
                    #print('w',width_size)
                    cropped_image = cv2.resize(cropped_image,(width_size, 128))
                    background[:cropped_image.shape[0], int(64 - width_size / 2):int(64 - width_size / 2)+ cropped_image.shape[1]] = cropped_image

                # if _w > _h:
                #     height_size = 90 * _h / _w
                #     #print('h',height_size)
                #     cropped_image = cv2.resize(cropped_image,(90, height_size))
                #     background[int(64 - height_size / 2):int(64 - height_size / 2)+ cropped_image.shape[0],19:19+ cropped_image.shape[1]] = cropped_image
                # else:
                #     width_size = 90 * _w / _h
                #     #print('w',width_size)
                #     cropped_image = cv2.resize(cropped_image,(width_size, 90))
                #     background[19:19+cropped_image.shape[0], int(64 - width_size / 2):int(64 - width_size / 2)+ cropped_image.shape[1]] = cropped_image


                # Processing to make the writing clearer
                img_arr = Image.fromarray(background)
                enhancerimg = ImageEnhance.Contrast(img_arr)
                _image = enhancerimg.enhance(1.5)
                opencv_image = np.array(_image)
                opencv_image = cv2.bilateralFilter(opencv_image, 9, 30, 30)


                cv2.imshow('back',opencv_image)
                cv2.waitKey(1)

                # Save image
                name = dst_dir + '/uni' + charList[idx] + '.png'
                cv2.imwrite(name, opencv_image)
                #cv2.imwrite('/home/yejin/PycharmProjects/FONTS/template_YJ/output_YJ/uni' + charList[idx] + '.png', opencv_image)

                idx += 1

                if idx == len(charList):
                    return

parser = argparse.ArgumentParser(description='Crop scanned images to character images')
parser.add_argument('--src_dir', dest='src_dir', required=True, help='directory to read scanned images')
parser.add_argument('--dst_dir', dest='dst_dir', required=True, help='directory to save character images')
args = parser.parse_args()


if __name__ == '__main__':
    #crop('/home/malab2/PycharmProjects/FONTS/template_YJ')
    crop_image_uniform(args.src_dir, args.dst_dir)

