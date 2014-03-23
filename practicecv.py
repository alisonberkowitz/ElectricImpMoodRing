import cv2
import numpy as np
import time
i=0

cap = cv2.VideoCapture(0)
_,fimg=cap.read()
cascade = cv2.CascadeClassifier("/home/epierce/Documents/haarcascade_frontalface_alt.xml")
def detect(path):
    #img = cv2.imread(path)
    img=path
#    cascade = cv2.CascadeClassifier("/home/epierce/Documents/haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.05, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    #cv2.imwrite('/home/epierce/Documents/detected.jpg', img);

# rects, img = detect("/home/epierce/Documents/faces.jpg")
# box(rects, img)
while(1):
    _,f = cap.read()
    if i%5==0:
        rects, img = detect(f)
        box(rects, img)
        cv2.imshow("Video",img)
    i=i+1
    key = cv2.waitKey(20)

    if key == 27:
        break
 
cv2.destroyAllWindows()
cv2.release()