import cv2
from numpy import true_divide 
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
import socket
import imagezmq
from save_to_csv import create, check_date_csv
import sys
sys.path.append("../..")
from Kien.milvus_image import Collection_Image

# def scrfd_box(result,path):
#     frame=path
#     # frame=cv2.resize(frame,[480,640])
#     face=SCRFD(result)
#     face.prepare()
#     bboxes, kpss = face.detect(path, 0.5, input_size = (640, 640))
#     for i in range(bboxes.shape[0]):
#         bbox = bboxes[i]
#         x1,y1,x2,y2,score = bbox.astype(np.int)
#         cv2.rectangle(path, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
#         if kpss is not None:
#             kps = kpss[i]
#             for kp in kps:
#                 kp = kp.astype(np.int)
#             cv2.circle(path, tuple(kp) , 1, (0,0,255) , 2)
#         path = path[y1:y2,x1:x2]
#         if path.shape <= (128, 128, 3):
#             path=frame
#         else:
#             path=path
#     return path
con =Collection_Image()
def Anti(result):
    if result == 1:
        real = True    
    else:
        real = False
    return real

def Mask(result):
    if result[0] >= 0.5:
        #no mask 
        mask=False
    else:
        #mask 
        mask=True
    return mask
    
def render_mark(face):
    tool = MaskRenderer()
    tool.prepare(det_size=(128, 128))
    params = tool.build_params(face)
    mask_out = tool.render_mask(face, 'mask_blue', params)# use single thread to test the time cost
    return mask_out

def cameraStream(url,connect_to):
    connect_to=connect_to
    cap=cv2.VideoCapture(url)
    sender = imagezmq.ImageSender(connect_to=connect_to)
    host_name = socket.gethostname()
    local_ip=socket.gethostbyname(host_name)
    cam_id='Camera 1'
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    while True:
        ret,frame1=cap.read()
        frame2=cv2.resize(frame1,[640,640])
        model=Flipface(frame2)
        
        scrfd=model.scrfd_out()
        face=SCRFD(scrfd)
        face.prepare()
        bboxes, kpss = face.detect(frame2, 0.6, input_size = (640, 640))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox.astype(np.int)
            cv2.rectangle(frame2, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int)
                cv2.circle(frame2, tuple(kp) , 1, (0,0,255) , 2)
            # face = frame2[y1:y2,x1:x2]
            # face=cv2.resize(face,(640,640))
            # print(face.shape)
            # if face.shape <= (128, 128, 3):
            #     frame=frame2
            # else:
            #     frame=face
                frame=frame2
                result_Mask=model.Mask(frame)
                mask=Mask(result_Mask)

                anti=Anti(model.AntiSpoofing())

                if mask == False :
                    Embeding_Face=model.Embeding_Face(frame)
                    #face_mask=render_mark(frame)
                    #Embeding_mask=model.Embeding_Face_mask(face_mask)
                    name=con.search_vectors(Embeding_Face,1)
                    #cv2.putText(frame,str(name["name"]),(x2-30,y2-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                    # if name["flag"]== True:
                    #     print(name["name"])
                    #     create()
                    #     check_date_csv(name['name'])
                    # else:
                    #     print(name["name"])
                    print(name)
                else : 
                    Embeding_mask=model.Embeding_Face_mask(frame)
                    name=con.search_vectors(Embeding_mask,1)
                    #cv2.putText(frame,str(name["name"]),(x2-30,y2-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                    # if name["flag"]== True:
                    #     print(name["name"])
                    print(name)
                    # else:
                    #     print(name["name"])
                    #     create()
                        # check_date_csv(name['name'])
        start= time.time()
        data={
            "id":local_ip,
            "cam":cam_id,
            "time": start
        }
        sender.send_image(data,frame2)

if __name__=='__main__':
    # path='2.jpg'
    # img=cv2.imread('2.jpg')
    connect_to='tcp://localhost:5555'
    url='rtsp://admin:lifeteK$1368@192.168.1.65/1'
    cap=cv2.VideoCapture(url)
    sender = imagezmq.ImageSender(connect_to=connect_to)
    host_name = socket.gethostname()
    local_ip=socket.gethostbyname(host_name)
    cam_id='Camera 1'
    img_counter = 0
  
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    framei=1
    while True:
        ret,frame1=cap.read()
        frame2=cv2.resize(frame1,[640,640])
        model=Flipface(frame2)
        
        scrfd=model.scrfd_out()
        face=SCRFD(scrfd)
        face.prepare()
        bboxes, kpss = face.detect(frame2, 0.6, input_size = (640, 640))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2,score = bbox.astype(np.int)
            cv2.rectangle(frame2, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int)
                cv2.circle(frame2, tuple(kp) , 1, (0,0,255) , 2)
            face = frame2[y1:y2,x1:x2]
            face=cv2.resize(face,(640,640))
            print(face.shape)
            if face.shape <= (128, 128, 3):
                frame=frame2
            else:
                frame=face
                #frame=frame2
                result_Mask=model.Mask(frame)
                mask=Mask(result_Mask)

                anti=Anti(model.AntiSpoofing())

                if mask == False :
                    Embeding_Face=model.Embeding_Face(frame)
                    #face_mask=render_mark(frame)
                    #Embeding_mask=model.Embeding_Face_mask(face_mask)
                    name=con.search_vectors(Embeding_Face,1)
                    #cv2.putText(frame,str(name["name"]),(x2-30,y2-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                    # if name["flag"]== True:
                    #     print(name["name"])
                    #     create()
                    #     check_date_csv(name['name'])
                    # else:
                    #     print(name["name"])
                    print(name)
                else : 
                    Embeding_mask=model.Embeding_Face_mask(frame)
                    name=con.search_vectors(Embeding_mask,1)
                    #cv2.putText(frame,str(name["name"]),(x2-30,y2-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                    # if name["flag"]== True:
                    #     print(name["name"])
                    print(name)
                    # else:
                    #     print(name["name"])
                    #     create()
                        # check_date_csv(name['name'])
        start= time.time()
        data={
            "id":local_ip,
            "cam":cam_id,
            "time": start
        }
        
        # if framei%5==0:
        #     continue
        # else:
        #     sender.send_image(data,frame2)
        # framei+=1
        sender.send_image(data,frame2)
