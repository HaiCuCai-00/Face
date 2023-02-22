from doctest import Example
from builtins import Exception, print
import base64
import json
from re import I
from tkinter import image_types
import uuid
import cv2
import numpy as np
from loguru import logger
import sys
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Type
import pydantic
from pydantic import BaseModel
import uvicorn
from insightface.app import MaskRenderer
sys.path.append("..")
from module.milvus_helpers import MilvusHelper
from module.operations.count import do_count
from module.operations.delete import do_delete
from module.operations.load import do_load
from module.operations.search import do_search
from module.operations.drop import do_drop
from module.operations.show import do_show
from module.mysql_helpers import MySQLHelper
from models.FACE_Triton.client_triton import Flipface
from models.FACE_Triton.Pose_processing.face import SCRFD
from models.FACE_Triton.face_detection import render_mark, face_analysis
#from configs.config import SCORE
import time
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(
        default=None, example=None, description="List of base64 encoded images"
    )
    urls: Optional[List[str]] = pydantic.Field(
        default=None, example=None, description="List of images urls"
    )
    person_id: Optional[List[str]] = pydantic.Field(
        default=None,
        example=None,
        description="List of person_id which use for enrollment",
    )
    table_name: str = pydantic.Field(
        default=None, example=None, description="Table name"
    )


milvus_cli=MilvusHelper()
mysql_cli=MySQLHelper()


@app.post("/infer")
async def infer(param:Images):
    try:
        s = time.time()
        if param.data is not None:
            image=param.data[0]
            image_bytes=base64.b64decode(image)
            img=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            try:
                kps = face_analysis(img)[0]['kps']
                if kps[0][0] > kps[2][0] and kps[1][0] > kps[2][0]:
                    img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif kps[0][0] < kps[2][0] and kps[1][0] < kps[2][0]:
                    img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
                else:
                    print("no rotate")
            except:
                print("Total: ", time.time() - s )
                logger.error("Can't rotate image face")
                return {"status": False, "msg": "Dữ liệu hình ảnh bị ngang"}
            img=cv2.resize(img,(640,640))
        else:
            logger.error("Not find input data")
            return {"status": False, "msg":"Không có ảnh"}
        model=Flipface(img)
        anti=model.AntiSpoofing()
        print("antispoofing: ",anti)
        if anti==1:
            scrfd=model.scrfd_out()
            face=SCRFD(scrfd)
            face.prepare()
            bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
            print('box ',bboxes.shape[0] )

            #print("box",bboxes)
            if bboxes.shape[0]==0:
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
            elif bboxes is not None:
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    x1,y1,x2,y2,score = bbox.astype(int)
                    # cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
                    if kpss is not None:
                        kps = kpss[i]
                        for kp in kps:
                            kp = kp.astype(int)
                        cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
                    img_face = img[y1:y2,x1:x2]
                    cv2.imwrite('anh_test.jpg',img_face)
                    mask=model.Mask(img_face)
                    print("mask: ",mask)
                    if mask[0]>=0.8:
                        Embeding_Face=model.Embeding_Face(img_face)
                        return {"embeding":Embeding_Face.tolist(), "type":"face"}
                    else:
                        Embeding_mask=model.Embeding_Face_mask(img_face)
                        return {"embeding": Embeding_mask.tolist(), "type": "face_mask"}
            else:
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
            
        else:
            logger.error("Fake face")
            return {"status": False, "msg":"Mặt giả"}

    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": e}
#1
@app.post("/count")
async def count(param:Images):
    try:
        if param.table_name is not None:
            num=do_count(param.table_name, milvus_cli)
            if num == None:
               return {"status": False, "msg": "Không tìm thấy bẳng"}
            logger.success("Successfully count the number of images in the database!")
            return {"status": True, "count": int(num/2) }
        else:
            return {"status": False, "msg": "Tên bảng không được phép"}
    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": str(e)}

#2
@app.post("/drop")
async def drop(param:Images):
    try:
        if param.table_name is not None:
            status=do_drop(param.table_name, milvus_cli,mysql_cli)
            #return status
            if status == "ok":
                return {"status": True, "msg":"Xóa khuôn mặt thành công"}
            else:
                return {"status": False, "msg":"Tên bảng không hợp lệ"}
        else:
            return {"status": False, "msg": "Tên bảng không được phép"}
    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": str(e)}

@app.post("/insert")
async def insert(param:Images):
    s = time.time()

    try:
        if param.data[0] is not None:
            print("11111", param.urls)
            print("22222", param.table_name)
            try:
                
                image=param.data[0]
                person_id=param.person_id[0]
                if person_id is None:
                    print("Missing person_id")
                    return {"status": False, "msg": "Thiếu dữ liệu nhân sự"}
                image_bytes=base64.b64decode(image)
                img=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                file_id = uuid.uuid1().hex
                file_path_hai = f"insert_f/{file_id}.jpg"
                cv2.imwrite(file_path_hai,img)
                try:
                    kps = face_analysis(img)[0]['kps']
                    if kps[0][0] > kps[2][0] and kps[1][0] > kps[2][0]:
                        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif kps[0][0] < kps[2][0] and kps[1][0] < kps[2][0]:
                        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
                    else:
                        print("no rotate")
                except:
                    print("Total: ", time.time() -s )
                    logger.error("Can't rotate image face")
                    return {"status": False, "msg": "Dữ liệu hình ảnh bị ngang"}
                img=cv2.resize(img,(640,640))
            except Exception as e:
                print("Total: ", time.time() -s )
                logger.error("Image data not found",e)
                return {"status": False, "msg": "Lỗi mã hóa"}
        else:
            print("Total: ", time.time() -s )
            logger.error("Not find input data")
            return {"status": False, "msg":"Không tìm thấy ảnh"}
        
        if person_id == "":
            print("Total: ", time.time() -s )
            return { "status": False, "msg": "id trống" }
        
        file_id = uuid.uuid1().hex
        image_path=f'static/{file_id}_{person_id}.jpg'
        image_path1=f'insert_p/{file_id}_{person_id}.jpg'
        mask_image_path=f'static/{file_id}_{person_id}_masked.jpg'

        try:
            save_face_mask=render_mark(img)
            cv2.imwrite(image_path,img)
            # code Hai test <
            cv2.imwrite(image_path1,img)
            # code Hai test >
            cv2.imwrite(mask_image_path,save_face_mask)
            print("save done: ",image_path)
        except Exception as e:
            logger.error("not find input face")
            print("Total: ", time.time() -s )
            return{"status": False, "msg":"Không thấy khuôn mặt"}
        
        model=Flipface(img)
        # anti=model.AntiSpoofing()
        # if anti==1:
        scrfd=model.scrfd_out()
        face=SCRFD(scrfd)
        face.prepare()
        bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
        #print("box",bboxes)
        if bboxes.shape[0]==0:
            print("Total: ", time.time() -s )
            logger.error("Not face")
            return {"status": False, "msg": "Không thấy khuôn mặt"}
        elif bboxes is not None:
             for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    x1,y1,x2,y2,score = bbox.astype(int)
                    # cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
                    if kpss is not None:
                        kps = kpss[i]
                        for kp in kps:
                            kp = kp.astype(int)
                        cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
                    img_face = img[y1:y2,x1:x2]
                    mask=model.Mask(img_face)
                    print(mask)
                    if mask[0]>=0.75:
                        Embeding_Face=model.Embeding_Face(img_face)
                        Embeding_Face=Embeding_Face.tolist()
                        print("path image",image_path)
                        face_id=do_load(param.table_name,[Embeding_Face,[False]],param.person_id,image_path,milvus_cli,mysql_cli)
                        face_mask=render_mark(img_face)
                        Embedding_mask=model.Embeding_Face_mask(face_mask)
                        Embedding_mask=Embedding_mask.tolist()
                        mask_id=do_load(param.table_name,[Embedding_mask,[True]],param.person_id,mask_image_path,milvus_cli,mysql_cli)
                        logger.info(f"Masked id {mask_id}, Non mask id {face_id}")

                        print("Total: ", time.time() -s )
                        return {"status": True, "msg":"Chèn thành công khuôn mặt vào cơ sở dữ liệu"}
                    else:
                        print("Total: ", time.time() -s )
                        return {"status": False,"msg":"Không chèn được khuôn mặt với khẩu trang"}
               
             else:
                print("Total: ", time.time() -s )
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
        # else:
        #     print("Total: ", time.time() -s )
        #     logger.error("Fake face")
        #     return {"status": False, "msg":"Mặt giả"}

    except Exception as e:
        print("Total: ", time.time() -s )
        logger.error(e)
        return {"status": False, "msg": str(e)}

@app.post("/recognize")
async def search(param:Images):
    try:
        
        if param.data is not None:
            try:
                start=time.time()
                image=param.data[0]
                # print("<<<<<>>>>", param.urls[0])
                # print("<<<<<>>>>", param.table_name[0])
                image_bytes=base64.b64decode(image)
                img=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                try:
                    kps = face_analysis(img)[0]['kps']
                    if kps[0][0] > kps[2][0] and kps[1][0] > kps[2][0]:
                        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif kps[0][0] < kps[2][0] and kps[1][0] < kps[2][0]:
                        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
                except:
                    logger.error("Can't rotate image face")
                    return {"status": False, "msg": "Dữ liệu hình ảnh bị ngang"}
                file_id = uuid.uuid1().hex
                file_path = f"upload/{file_id}.jpg"
                print("upload", file_id)
                cv2.imwrite(file_path,img)
                img=cv2.resize(img,(640,640))
                img_f_id = img.copy()
                
            except Exception as e:
                logger.error("Image data not found")
                return {"status": False, "msg": "Dữ liệu hình ảnh không được tìm thấy"}
        else:
            logger.error("Not find input data image")
            return {"status":False, "msg":"Không tìm thấy hình ảnh dữ liệu đầu vào"}
        
        model=Flipface(img)
        # bỏ anti thì sẽ ổn định hơn 
        # anti=model.AntiSpoofing()
        # if anti==1:
        scrfd=model.scrfd_out()
        face=SCRFD(scrfd)
        face.prepare()
        bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
        print("box",bboxes)
        if bboxes.shape[0]==0:
            logger.error("Not face")
            return {"status": False, "msg": "Không thấy khuôn mặt"}
        print("box", bboxes)
        if bboxes is not None:
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                x1,y1,x2,y2,score = bbox.astype(np.int)
                #cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
                
                if kpss is not None:
                    kps = kpss[i]
                    for kp in kps:
                        kp = kp.astype(np.int)
                    cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
                img_face = img[y1:y2,x1:x2]
                mask=model.Mask(img_face)
                if mask[0]>=0.8:
                    Embeding_Face=model.Embeding_Face(img_face)
                    Embeding_Face=Embeding_Face.tolist()
                    person_id, distance, path_image=do_search(param.table_name,Embeding_Face,1,False,milvus_cli,mysql_cli)
                    end=time.time()
                    print("time: ",end-start)
                else:
                    Embeding_mask=model.Embeding_Face_mask(img_face)
                    Embeding_mask=Embeding_mask.tolist()
                    person_id, distance, path_image=do_search(param.table_name,Embeding_mask,1,True,milvus_cli,mysql_cli)
                print("id",person_id)
                print("distance", distance)
                if distance is None:
                    return {"status": True,"msg": "Người lạ"}
                elif distance[0]<= 400:
                    return {"status": True, "person_id":person_id[0], "path_image": path_image}
                else:
                    return {"status": True, "msg": "Người lạ"}
        else:
            return {"status": False, "msg":"Không thấy khuôn mặt"}
        # else:
        #     return {"status": False,"msg":"Mặt giả"}
    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": "Không thấy khuôn mặt"}
#3
@app.post("/delete")
async def delete(param:Images):
    try:
        person_id = param.person_id
        if person_id is not None:
            status=do_delete(param.table_name, person_id, milvus_cli, mysql_cli)
            logger.info("Successfully delete entity in Milvus and MySQL!")
            if status=="ok":
                return {"status": True, "msg": "Xóa thành công"}
            else:
                return {"status": False,"msg": "id không hợp lệ"}
        else:
            return {"status": False, "msg":"Tên bảng trống"}
    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": str(e)}

@app.post("/showimage")
async def show(param:Images):
    try:
        person_id=param.person_id
        table_name= param.table_name
        if person_id is not None:
            res=do_show(table_name, person_id,mysql_cli,milvus_cli)
            if res =="no_table":
                return {"status": False, "msg": "id không hợp lệ"}
            else:
                return {"status": True, "path_image": res}
          
    except Exception as e:
        logger.error(e)
        return{"status": False, "msg": "Người dùng không có ảnh"}
@app.post("/insert0")
async def insert(param:Images):
    try:
        if param.data[0] is not None:
            try:
                image=param.data[0]
                person_id=param.person_id[0]
                image_bytes=base64.b64decode(image)
                img=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                img=cv2.resize(img,(640,640))
                
            except Exception as e:
                logger.error("Image data not found")
                return {"status": False, "msg": "Lỗi mã hóa"}
        else:
            logger.error("Not find input data")
            return {"status": False, "msg":"Không tìm thấy ảnh"}
        
        if person_id == "":
            return { "status": False, "msg": "id trống" }
        
        file_id = uuid.uuid1().hex
        image_path=f'static/{file_id}_{person_id}.jpg'
        mask_image_path=f'static/{file_id}_{person_id}_masked.jpg'

        try:
            save_face_mask=render_mark(img)
            cv2.imwrite(image_path,img)
            cv2.imwrite(mask_image_path,save_face_mask)
            print("save done")
        except Exception as e:
            logger.error("not find input face")
            return{"status": False, "msg":"Không thấy khuôn mặt"}
        
        model=Flipface(img)
        anti=model.AntiSpoofing()
        if anti==1:
            scrfd=model.scrfd_out()
            face=SCRFD(scrfd)
            face.prepare()
            bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
            #print("box",bboxes)
            if bboxes.shape[0]==0:
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
            elif bboxes is not None:
                
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    x1,y1,x2,y2,score = bbox.astype(np.int)
                    # cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
                    if kpss is not None:
                        kps = kpss[i]
                        for kp in kps:
                            kp = kp.astype(np.int)
                        cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
                    img_face = img[y1:y2,x1:x2]
                    mask=model.Mask(img_face)
                    print(mask)
                    if mask[0]>=0.75:
                        Embeding_Face=model.Embeding_Face(img_face)
                        Embeding_Face=Embeding_Face.tolist()
                        print("path image",image_path)
                        face_id=do_load(param.table_name,[Embeding_Face,[False]],param.person_id,image_path,milvus_cli,mysql_cli)
                        face_mask=render_mark(img_face)
                        Embedding_mask=model.Embeding_Face_mask(face_mask)
                        Embedding_mask=Embedding_mask.tolist()
                        mask_id=do_load(param.table_name,[Embedding_mask,[True]],param.person_id,mask_image_path,milvus_cli,mysql_cli)
                        logger.info(f"Masked id {mask_id}, Non mask id {face_id}")

                        return {"status": True, "msg":"Chèn thành công khuôn mặt vào cơ sở dữ liệu"}
                    else:
                        return {"status": False,"msg":"Không chèn được khuôn mặt với khẩu trang"}
            else:
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
        else:
            logger.error("Fake face")
            return {"status": False, "msg":"Mặt giả"}

    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": str(e)}

@app.post("/recognize0")
async def search(param:Images):
    try:
        if param.data is not None:
            try:
                start=time.time()
                
                image=param.data[0]
                image_bytes=base64.b64decode(image)
                img=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                img=cv2.resize(img,(640,640))
                cv2.imwrite("anh.jpg",img)
            except Exception as e:
                logger.error("Image data not found")
                return {"status": False, "msg": "Dữ liệu hình ảnh không được tìm thấy"}
        else:
            logger.error("Not find input data image")
            return {"status":False, "msg":"Không tìm thấy hình ảnh dữ liệu đầu vào"}
        
        model=Flipface(img)
        anti=model.AntiSpoofing()
        if anti==1:
            scrfd=model.scrfd_out()
            face=SCRFD(scrfd)
            face.prepare()
            bboxes, kpss = face.detect(img, 0.6, input_size = (640, 640))
            print("box",bboxes)
            if bboxes.shape[0]==0:
                logger.error("Not face")
                return {"status": False, "msg": "Không thấy khuôn mặt"}
            print("box", bboxes)
            if bboxes is not None:
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    x1,y1,x2,y2,score = bbox.astype(np.int)
                    cv2.rectangle(img, (x2,y2)  , (x1,y1) , (255,0,0) , 2)
                    
                    if kpss is not None:
                        kps = kpss[i]
                        for kp in kps:
                            kp = kp.astype(np.int)
                        cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
                    img_face = img[y1:y2,x1:x2]
                    mask=model.Mask(img_face)
                    if mask[0]>=0.8:
                        print("zzzzzzzzzzzz")
                        Embeding_Face=model.Embeding_Face(img_face)
                        Embeding_Face=Embeding_Face.tolist()
                        person_id, distance, path_image=do_search(param.table_name,Embeding_Face,1,False,milvus_cli,mysql_cli)
                        end=time.time()
                        print("time: ",end-start)
                    else:
                        print("wwwwwwwwwww")
                        Embeding_mask=model.Embeding_Face_mask(img_face)
                        Embeding_mask=Embeding_mask.tolist()
                        person_id, distance, path_image=do_search(param.table_name,Embeding_mask,1,True,milvus_cli,mysql_cli)
                    print("id",person_id)
                    print("distance", distance)
                    if distance is None:
                            return {"status": True,"msg": "Người lạ"}
                    elif distance[0]<= 400:
                        return {"status": True, "person_id":person_id[0], "path_image": path_image}
                    else:
                        return {"status": True, "msg": "Người lạ"}
            else:
                return {"status": False, "msg":"Không thấy khuôn mặt"}
        else:
            return {"status": False,"msg":"Mặt giả"}
    except Exception as e:
        logger.error(e)
        return {"status": False, "msg": "Không thấy khuôn mặt"}
#3
if __name__=="__main__":
    uvicorn.run(app, host='0.0.0.0', port=8888)
