import os
import cv2
import numpy as np

# must import if working with opencv in python
import numpy as np
import cv2

def chunksList(List,size):
    for i in range(0, len(List), size):
        yield List[i : i + size]


def crop (img_dir):
    imgList = os.listdir(img_dir)
    for img in imgList:
        image = cv2.imread(os.path.join(img_dir,img))



# template size 280 X 200 mm
# space size    1.5 X 1.5 mm

#image = cv2.imread('/home/malab2/PycharmProjects/FONTS/image/template.png')
image = cv2.imread('/home/malab2/PycharmProjects/FONTS/image/1.jpg')

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
H,W = gray.shape


cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.imshow('gray', gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)

cv2.namedWindow('second', cv2.WINDOW_NORMAL)
cv2.imshow('second', thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((13,13), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.namedWindow('dilated', cv2.WINDOW_NORMAL)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


cnt = 0
rect = []

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box

    x, y, w, h = cv2.boundingRect(ctr)

    print("H", h / float(H), "w", w / float(W))

    # Getting ROI
    roi = image[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)

    if h / float(H) >= 0.045 and w / float(W) >= 0.065:
        rect.append([x,y,w,h])
        #cv2.rectangle(image, (x, y), (x + w, y + h), (171, 255, 0), 3)
        #cv2.namedWindow('marked areas', cv2.WINDOW_NORMAL)
        #cv2.imshow('marked areas', image)
        #cv2.waitKey(0)
        cnt = cnt+1

del rect[0]
print(len(rect))
rect = sorted(rect , key=lambda k: [k[1]])
rect = list(chunksList(rect,12))


idx = 0
f = open('399-uniform.txt','r')
charList = f.readlines()

for i in range(len(charList)):
    charList[i] = charList[i].strip()

# case2
# charList = [i.strip() for i in charList]



for row in rect:
    print(row)
    i = sorted(row, key=lambda k: [k[0]])
    for rec in row:
        x, y, w, h = rec
        roi = image[y:y + h + 1, x:x + w + 1] # 1 is margin

        cv2.namedWindow('marked areas', cv2.WINDOW_NORMAL)
        cv2.imshow('marked areas', roi)
        cv2.waitKey(10)

        cv2.imwrite('/home/malab2/PycharmProjects/FONTS/output/'+charList[idx]+'.png', roi)

        idx += 1

        # cv2.imwrite('/home/malab2/PycharmProjects/FONTS/output/test.png', roi)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # cv2.namedWindow('marked areas', cv2.WINDOW_NORMAL)
        # cv2.imshow('marked areas', image)
        # cv2.waitKey(1)






