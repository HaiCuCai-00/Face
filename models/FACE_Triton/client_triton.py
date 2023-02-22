import argparse
import numpy as np
import sys
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import torch.nn as nn 

class Flipface:
    def __init__(self,images,
        triton_uri="localhost:8001",
        verbose = False, 
        ssl= False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None
        ):
        self.image=images
        self.triton_uri = triton_uri
        self.verbose = verbose
        self.ssl = ssl
        self.root_certificates=root_certificates
        self.private_key=private_key
        self.certificate_chain=certificate_chain

        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_uri,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)
    
    def scrfd_out(self,model_name='scrfd'):
        inputs=[]
        outputs=[]
        inputs.append(grpcclient.InferInput('input.1', [1, 3, 640, 640], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('score_8'))
        outputs.append(grpcclient.InferRequestedOutput('bbox_8'))
        outputs.append(grpcclient.InferRequestedOutput('score_16'))
        outputs.append(grpcclient.InferRequestedOutput('bbox_16'))
        outputs.append(grpcclient.InferRequestedOutput('score_32'))
        outputs.append(grpcclient.InferRequestedOutput('bbox_32'))

        input_image_buffer = cv2.resize(self.image, [640, 640])
        #frame = cv2.resize(self.image,[480,640])
        input_image_buffer = cv2.cvtColor(input_image_buffer, cv2.COLOR_BGR2RGB)
        input_image_buffer = input_image_buffer.transpose((2, 0, 1)).astype(np.float32)
        input_image_buffer /= 255.0
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        # Send data to server 
        inputs[0].set_data_from_numpy(input_image_buffer)

        results = self.triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)
    
        detections=[]
        detections.append(results.as_numpy('score_8'))
        detections.append(results.as_numpy('bbox_8'))
        detections.append(results.as_numpy('score_16'))
        detections.append(results.as_numpy('bbox_16'))
        detections.append(results.as_numpy('score_32'))
        detections.append(results.as_numpy('bbox_32'))            
        return detections

    def AntiSpoofing(self,model_name='Anti_Mini_V2'):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('modelInput', [1, 3, 80, 80], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('modelOutput'))
    
        input_image_buffer = cv2.resize(self.image, [80, 80])
        input_image_buffer = input_image_buffer.transpose((2, 0, 1)).astype(np.float32)
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)
 
        results = self.triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

        lable = results.as_numpy('modelOutput')
        result = np.argmax(lable)
    
        return result
    
    def Mask(self,embed_img,model_name='Mask'):
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('conv2d_input', [1, 3, 128, 128], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('dense_1')) 
        
        input_image_buffer = cv2.resize(embed_img, [128, 128])
        input_image_buffer = cv2.cvtColor(input_image_buffer, cv2.COLOR_BGR2RGB)
        input_image_buffer = input_image_buffer.transpose((2, 0, 1)).astype(np.float32)
        input_image_buffer /= 255.0
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)
        
        results = self.triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

        result = results.as_numpy('dense_1')

        return result
    
    def Embeding_Face(self,embed_img,model_name='R50'):

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('modelInput', [1, 3, 112, 112], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('modelOutput')) 
        
        input_image_buffer = cv2.resize(embed_img, [112, 112])
        input_image_buffer = cv2.cvtColor(input_image_buffer, cv2.COLOR_BGR2RGB)
        input_image_buffer = input_image_buffer.transpose((2, 0, 1)).astype(np.float32)
        input_image_buffer /= 255.0
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)
        
        results = self.triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

        result = results.as_numpy('modelOutput')

        return result
    
    def Embeding_Face_mask(self,embed_img,model_name='Embed_mask'):

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('modelInput', [1, 3, 112, 112], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('modelOutput')) 
        input_image_buffer = cv2.resize(embed_img, [112, 112])
        input_image_buffer = cv2.cvtColor(input_image_buffer, cv2.COLOR_BGR2RGB)
        input_image_buffer = input_image_buffer.transpose((2, 0, 1)).astype(np.float32)
        input_image_buffer /= 255.0
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)
        
        results = self.triton_client.infer(model_name=model_name,
                                inputs=inputs,
                                outputs=outputs,
                                client_timeout=None)

        result = results.as_numpy('modelOutput')

        return result

if __name__=="__main__":
    img=cv2.imread("/media/ai-r-d/DATA1/Face_triton/models/FACE_Triton/3.jpg")
    model=Flipface(img)
    anti=model.Embeding_Face(img)
    print(anti)
    
