from Pose_processing.face import SCRFD
from client_triton import Flipface
import os, sys, datetime
import numpy as np
import os.path as osp
import cv2

def scrfd_box1(result,path):
    img=cv2.imread(path)
    face=SCRFD(result)
    face.prepare()
    bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(np.int)
        point = (x1,y1,x2,y2)
    return point

def scrfd_box(result,path):
    img=cv2.imread(path)
    model=Flipface(img)
    scrfd=model.scrfd_out()
    face=SCRFD(scrfd)
    face.prepare()
    bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
    print("boxs:",bboxes)
    print(kpss)

if __name__ == "__main__":
    path='/media/ai-r-d/DATA1/Face_triton/models/FACE_Triton/images.jpeg'

    img=cv2.imread('/media/ai-r-d/DATA1/Face_triton/models/FACE_Triton/images.jpeg')

    model=Flipface(img)

    scrfd=model.scrfd_out()

    face=scrfd_box(scrfd,path)