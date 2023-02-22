import cv2 
from client_triton import Flipface
import os, sys, datetime
import numpy as np
import os.path as osp
import cv2
from Pose_processing.face import SCRFD
import insightface
from insightface.app import MaskRenderer
from insightface.data import get_image as ins_get_image
import time  

def scrfd_box(result,path):
    img=cv2.imread(path)
    face=SCRFD(result)
    face.prepare()
    bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
    print("box",bboxes)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1,y1,x2,y2,score = bbox.astype(np.int)
        # cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                kp = kp.astype(np.int)
            cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        img = img[y1:y2,x1:x2]
    return img

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

if __name__=='__main__':
    path='/media/DATA_Old/Face_triton/models/FACE_Triton/5.jpeg'

    img=cv2.imread('/media/DATA_Old/Face_triton/models/FACE_Triton/5.jpeg')
    
    model=Flipface(img)

    scrfd=model.scrfd_out()

    result_Mask=model.Mask(img)

    face=scrfd_box(scrfd,path)

    mask=Mask(result_Mask)
    anti=Anti(model.AntiSpoofing())

    print(anti)
    if mask == False :
        Embeding_Face=model.Embeding_Face(face)
        face_mask=render_mark(face)
        Embeding_mask=model.Embeding_Face_mask(face_mask)
        print(Embeding_Face,Embeding_mask)
        
    else :
        Embed_mask=model.Embeding_Face_mask(face)
        print(Embed_mask)
    #print(face)
    cv2.imwrite('31.jpg',face_mask)
    #print('done')

    


    



