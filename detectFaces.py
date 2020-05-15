import cv2
import numpy
import os


class FaceDetector:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detectFromImage(self, image):
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi_color = None
            faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
            faces_list = []
            for (x, y, w, h) in faces:
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]
                faces_list.append(roi_color)
            return faces_list
        except:
            pass

    def getFacesFromFolder(self, path_in, path_out):
        curr = 0
        imgsNames = os.listdir(path_in)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        for name in imgsNames:
            imgPath = os.path.join(path_in, name)
            faces = self.detectFromImage(cv2.imread(imgPath))
            if faces != None:
                for face in faces:
                    facePath = os.path.join(path_out, str(curr).__add__('.jpg'))

                    while os.path.exists(facePath):
                        curr = curr + 1
                        facePath = os.path.join(path_out, str(curr).__add__('.jpg'))
                    if face is not None:
                        curr = curr + 1
                        cv2.imwrite(facePath, face)
