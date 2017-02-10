__author__ = 'max'
import numpy as np
import cv2

classifier = "./classifier/"
img = "./img/"

lower_blue = np.array([242,26,26,255])
upper_blue = np.array([242,30,30,255])

hat = cv2.imread(img+'hat.png')
hat = cv2.resize(hat,(200,200))

# bluehat = cv2.inRange(hat , lower_blue, upper_blue)

# res = cv2.bitwise_and(hat, hat, mask=bluehat)
# hat = cv2.cvtColor(hat, cv2.COLOR_HSV2RGB)


tmp = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(hat)
rgba = [b,g,r, alpha]
hat = cv2.merge(rgba,4)




# banana_cascade = cv2.CascadeClassifier(folder+'banana_classifier.xml')
eyes_cascade = cv2.CascadeClassifier(classifier + 'face.xml')
face_cascade = cv2.CascadeClassifier(classifier + 'fullface.xml')

cap = cv2.VideoCapture(0)


while(1):

    # Take each frame
    _, frame = cap.read()


    # define range of blue color in HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, alpha1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    b1 ,g1, r1 = cv2.split(frame)
    rgba1 = [b1,g1,r1,alpha1]
    frame = cv2.merge(rgba1,4)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.6, 4)
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w1, y + y1 + h1), (255, 255, 0), 2)
            ##let put a hat on now

        xh1 = x
        xh2 = x + 200

        yh1 = y -200
        yh2 = y

        # dst = cv2.add(hat,frame)


        # print xh1
        # print xh1

        if not (xh1 < 0 or xh2 > frame.shape[1] or yh1 < 0 or yh2 > frame.shape[0]):
            frame[yh1:yh2,xh1:xh2] = hat






    #cv2.imshow("sup",hat)


    cv2.imshow('Smile!',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()