#take image path from mysql
#convert image to vector
#import vector to milvus and take the milvus_id
#change milvus_id in mysql

import sys
import cv2
import os
import numpy as np

sys.path.append("..")
from mysql_helpers import MySQLHelper
from milvus_helpers import MilvusHelper
sys.path.append("/media/DATA_Old/hai/lifetek_project/Face/Face_triton")
from models.FACE_Triton.client_triton import Flipface
from models.FACE_Triton.Pose_processing.face import SCRFD
from models.FACE_Triton.face_detection import render_mark

if __name__ == "__main__":
    mysql = MySQLHelper()
    milvus = MilvusHelper()
    collection_name ='_AI'
    path = "/media/DATA_Old/hai/lifetek_project/Face/Face_triton/service"
    image_paths = mysql.show_data(collection_name)
    count = 0
    for i in image_paths:
        count += 1
        if count%2 == 0:
            continue 
        image_path = os.path.join(path,i[1])
        img = cv2.imread(image_path)
        model=Flipface(img)
        anti=model.AntiSpoofing()
        if anti==1:
            scrfd=model.scrfd_out()
            face=SCRFD(scrfd)
            face.prepare()
            bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
            if bboxes.shape[0] !=0:
                for j in range(bboxes.shape[0]):
                    bbox = bboxes[j]
                    x1,y1,x2,y2,score = bbox.astype(int)
                    img_face = img[y1:y2,x1:x2]
                    mask=model.Mask(img_face)
                    # print(mask)
                    if mask[0]>=0.75:
                        Embeding_Face=model.Embeding_Face(img_face)
                        Embeding_Face=Embeding_Face.tolist()
                        id = milvus.insert(collection_name, [Embeding_Face,[False]] )[0]
                        print(id)
                        mysql.upadte_milvus_id(collection_name,id,i[1])
                        face_mask=render_mark(img_face)
                        Embedding_mask=model.Embeding_Face_mask(face_mask)
                        Embedding_mask=Embedding_mask.tolist()
                        mask_id = milvus.insert(collection_name, [Embeding_Face,[True]] )[0]
                        mask_path = i[1].replace(".jpg","_masked.jpg")
                        print(mask_id)
                        mysql.upadte_milvus_id(collection_name,mask_id,mask_path)
                    
        