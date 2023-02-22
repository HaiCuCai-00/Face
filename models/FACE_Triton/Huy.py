import cv2 
from client_triton import Flipface
import os, sys, datetime
import numpy as np
import os.path as osp

from Pose_processing.face import SCRFD
import insightface
from insightface.app import MaskRenderer
from insightface.data import get_image as ins_get_image
import time  

def Anti(result):
    print(result)
    if result == 1:
        real = True
    else:
        real = False
    return real
 
def Mask(result):
    if result[0] >= 0.8:
        #no mask 
        mask=False
    else:
        #mask 
        mask=True
    return mask
    
def render_mark(face):
    tool = MaskRenderer()
    tool.prepare(det_size=(128,128))
    #image = cv2.imread(face)
    params = tool.build_params(face)
    mask_out = tool.render_mask(face, 'mask_blue', params)# use single thread to test the time cost
    return mask_out

def benmark_mask():
    mask_path = '/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/data_H/mask'
    nomask_path = '/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/data_H/nomask/'

    sum = 0
    mask = 0
    nomask = 0
    s = time.time()

    for file_name in os.listdir(mask_path):
        file_path = os.path.join(mask_path,file_name)
        img = cv2.imread(file_path)
        img2 = cv2.resize(img,[640,640])
        model = Flipface(img2)
        result = Mask(model.Mask(img2))
        if result:
            mask += 1
        else:
            nomask += 1
        sum += 1
    TP = mask
    FN = nomask

    mask = 0
    nomask = 0
    
    for file_name in os.listdir(nomask_path):
        file_path = os.path.join(nomask_path,file_name)
        img = cv2.imread(file_path)
        img2 = cv2.resize(img,[640,640])
        model = Flipface(img2)
        result = Mask(model.Mask(img2))
        if result:
            mask += 1
        else:
            nomask += 1
        sum += 1
    FP = mask
    TN = nomask

    acc = (TP+TN)/sum*100
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2/((1/precision)+(1/recall))
    
    print('Tong so du doan dung: ', TP+TN,'/',sum)
    print('Time: ',round((time.time() - s),3))
    print('Accuracy:', round(acc,2),'%')
    print('So du doan dung mask: ',TP)
    print('So du doan dung nomask: ',TN)
    print('So du doan sai nomask: ',FP)
    print('So du doan sai mask: ',FN)
    print('F1 score:', round(F1,3))
       
def rendermask():
    path='/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/data_H/nomask/'
    name = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path,file_name)
        img1=cv2.imread(file_path)

        file_name = file_name.strip('.png')
        save_path = '/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/data_H/render_mask/'
        try:
            face_mask=render_mark(img1)
            cv2.imwrite(save_path+file_name+'.jpg',face_mask)
        except:
            name.append(file_name)

def scrfd_box(result,path):
    img=cv2.imread(path)
    face=SCRFD(result)
    face.prepare()
    bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(np.int)
        point = (x1,y1,x2,y2)
    return point

if __name__=='__main__':
    path='/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/5.jpeg'

    img=cv2.imread('/media/DATA_Old/TRT_docker/FACE_Triton_v2/FACE_Triton/5.jpeg')
    img=cv2.resize(img,[640,640])
    model=Flipface(img)

    scrfd=model.scrfd_out()

    face=scrfd_box(scrfd,path)
    print(face)