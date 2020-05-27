import cv2
eye_detector=cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)
    if len(faces)==1:
        for (x,y,w,h) in faces:
            ex=x+10
            ey=y+60
            ew=w-20
            eh=h-180
            mask = cv2.imread('Image/glasses.png',-1)
            mask=cv2.resize(mask,(ew,eh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            ew,eh,ec=mask.shape
            for i in range(0,ew):
                for j in range(0,eh):
                    if mask[i, j][3] != 0:
                        img[ey + i,ex + j] = mask[i, j]
            mask = cv2.imread('Image/hat.png',cv2.IMREAD_UNCHANGED)
            hx=x-40
            hy=y-130
            hw=w
            hh=h-100
            mask=cv2.resize(mask,(hw+70,hh+20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            hw,hh,hc=mask.shape
            for i in range(0,hw):
                for j in range(0,hh):
                    if mask[i, j][3] != 0:
                        img[hy + i,hx + j] = mask[i, j]
            lx=x+70
            ly=y+150
            lw=w
            lh=h
            mask = cv2.imread('Image/mustache.png',cv2.IMREAD_UNCHANGED)
            mask=cv2.resize(mask,(lw-120,lh-210))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            w,h,c=mask.shape
            for i in range(0,w):
                for j in range(0,h):
                    if mask[i, j][3] != 0:
                        img[ly + i,lx + j] = mask[i, j]
    cv2.imshow("Hat Goggles Mustache Filter",img)
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv2.imwrite("Hat_Goggles_Mustache_Filter.png",img)
cap.release()
cv2.destroyAllWindows()
