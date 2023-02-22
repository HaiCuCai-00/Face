import sys
sys.path.append("..")
from models.FACE_Triton.client_triton import Flipface
from models.FACE_Triton.Pose_processing.face import SCRFD
from module.operations.delete import move

import os, sys, datetime, time
import numpy as np
import os.path as osp
import cv2
# from models.FACE_Triton.face_detection import render_mark,face_analysis

from insightface.app import MaskRenderer,FaceAnalysis

def scrfd_box(result,path):
    frame=path
    # frame=cv2.resize(frame,[480,640])
    face=SCRFD(result)
    face.prepare()
    bboxes, kpss = face.detect(path, 0.5, input_size = (640, 640))
    print("boxs:",bboxes)
    print("kpss: ",kpss)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(int)
        cv2.rectangle(path, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                kp = kp.astype(int)
            cv2.circle(path, tuple(kp) , 1, (0,0,255) , 2)
        path = path[y1:y2,x1:x2]
        if path.shape <= (128, 128, 3):
            path=frame
        else:
            path=path
    return path
    
def face_analysis(face):
    model_name = 'buffalo_l'
    tool = FaceAnalysis(name= model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    tool.prepare(ctx_id=0, det_size=(640, 640))
    kps = tool.get(face)[0]['kps']
    print(kps)
    if kps[0][0] > kps[2][0] and kps[1][0] > kps[2][0]:
        print(1)
        img = cv2.rotate(face,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif kps[0][0] < kps[2][0] and kps[1][0] < kps[2][0]:
        print(2)
        img = cv2.rotate(face,cv2.ROTATE_90_CLOCKWISE)
    return img
if __name__ == "__main__":
    path='/media/ai-r-d/DATA1/Face_triton/service/deleted_images/b8dcbc2c5e7011ed9897875b5391b198_626a1ca115e49f66d33d5c8f.jpg'
    path2 = "/media/ai-r-d/DATA1/Face_triton/service/deleted_images/2c12be6655ad11ed983967624173304b_633554ad713bea0e8df560fc.jpg"
    img=cv2.imread(path)

    model=Flipface(img)

    scrfd=model.scrfd_out()

    face=SCRFD(scrfd)
    face.prepare()
    bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
    print(bboxes, kpss)
    if bboxes is not None:
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox.astype(int)
            #cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
            
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(int)
                cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
            img_face = img[y1:y2,x1:x2]
            mask=model.Mask(img_face)
            print(1)
            cv2.imwrite("2.jpg",img_face)
    else:
        print("Không thấy khuôn mặt")
    # cv2.imwrite("2.jpg",face)
    # person_id = '626a1ca215e49f66d33d5c9b'
    # move(person_id)