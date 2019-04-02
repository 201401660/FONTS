# -*- coding: utf-8 -*-

import os
# must import if working with opencv in python
import numpy as np
import cv2


def chunksList(List,size):
    for i in range(0, len(List), size):
        yield List[i : i + size]


# template size 280 X 200 mm
# space size    1.5 X 1.5 mm

def crop (img_dir):

    imgList = sorted(os.listdir(img_dir))

    idx = 0
    f = open('399-uniform.txt', 'r')
    charList = f.readlines()

    for i in range(len(charList)):
        charList[i] = charList[i].strip()  # strip : delete '\n'


    for img in imgList:
        image = cv2.imread(os.path.join(img_dir,img))

        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)

        # binary
        #blur = cv2.bilateralFilter(gray, 9, 75, 75)
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
        _, ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours - 안해도 상관없음
        #sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

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

        #print('selected rectangle:', rect)

        rect = list(chunksList(rect, 12))  # 12개(= 1 row )만큼씩 묶어줌

        # case2
        # charList = [i.strip() for i in charList]

        for row in rect:
            #print(row)
            row = sorted(row, key=lambda k: [k[0]])  # row별로 column(=x 좌표)에 대해 정렬
            for rec in row:

                x, y, w, h = rec
                roi = thresh[y:y + h + 1, x:x + w + 1]  # 1 is margin
                open_kernel = np.ones((15,15), np.uint8)
                dilated_image = cv2.dilate(roi, open_kernel, iterations=1)
                dilated_image = ~dilated_image
                _, contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                #_r = sorted([ cv2.contourArea(contour) for contour in enumerate(contours)])[-2]
                #_r = sorted([cv2.contourArea(r) for r in contours])[-2]
                _r = ([contour for contour in contours])[-2]
                _x, _y, _w, _h = cv2.boundingRect(_r)
                #img1 = cv2.rectangle(dilated_image, (_x, _y), (_x + _w, _y + _h), 7)
                #cv2.imshow('test',img1)

                #open_kernel = np.ones((2,2), np.uint8)
                #dilated_image = cv2.dilate(roi, open_kernel, iterations=1)
                #blur = cv2.bilateralFilter(roi, 9,75,75)
                #opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, open_kernel)
                #cv2.imshow('opening',img1)
                #cv2.waitKey(0)

                # resize according to width and height
                background = np.ones((128, 128)) * 255
                cropped_image = ~roi[_y:_y+_h, _x:_x+_w]
                if _w > _h:
                    height_size = 90 * _h / _w
                    print('h',height_size)
                    cropped_image = cv2.resize(cropped_image,(90, height_size))
                    print(cropped_image.shape[0], cropped_image.shape[1])
                    background[int(64 - height_size / 2):int(64 - height_size / 2)+ cropped_image.shape[0],19:19+ cropped_image.shape[1]] = cropped_image
                else:
                    width_size = 90 * _w / _h
                    print('w',width_size)
                    cropped_image = cv2.resize(cropped_image,(width_size, 90))
                    print(cropped_image.shape[0],cropped_image.shape[1])
                    background[19:19+cropped_image.shape[0], int(64 - width_size / 2):int(64 - width_size / 2)+ cropped_image.shape[1]] = cropped_image

                #result = cv2.bilateralFilter(background, 9, 75, 75)
                cv2.imshow('back',background)
                cv2.waitKey(1)

                cv2.imwrite('/home/yejin/PycharmProjects/font/output/uni' + charList[idx] + '.png', background)
                idx += 1

                if idx == 399:
                    return



if __name__ == '__main__':
    #crop('/home/malab2/PycharmProjects/FONTS/image')
    crop('/home/yejin/PycharmProjects/font/template')

