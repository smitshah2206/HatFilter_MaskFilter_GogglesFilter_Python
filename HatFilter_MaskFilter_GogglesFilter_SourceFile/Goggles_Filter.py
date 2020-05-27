import cv2
eye_detector=cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        ex=x+10
        ey=y+60
        ew=w-20
        eh=h-180
        mask = cv2.imread('Image/glasses.png',-1)
        mask=cv2.resize(mask,(ew,eh))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        w,h,c=mask.shape
        for i in range(0,w):
            for j in range(0,h):
                if mask[i, j][3] != 0:
                    img[ey + i,ex + j] = mask[i, j]
    cv2.imshow("Glass Image",img)
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv2.imwrite("Glass_Filter.png",img)
cap.release()
cv2.destroyAllWindows()
