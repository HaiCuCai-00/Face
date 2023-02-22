from insightface.app import MaskRenderer
import cv2

def render_mark(face):
    tool =  MaskRenderer("antelopev2", root="/media/ai-r-d/DATA1/Face_triton/weights")

    tool.prepare(det_size=(128,128))
    #image = cv2.imread(face)
    params = tool.build_params(face)
    mask_out = tool.render_mask(face, 'mask_blue', params)# use single thread to test the time cost
    return mask_out

def Mask(result):
    if result[0] >= 0.8:
        #no mask 
        mask=False
    else:
        #mask 
        mask=True
    return mask

def Anti(result):
    print(result)
    if result == 1:
        real = True    
    else:
        real = False
    return real

if __name__=="__main__":
    img=cv2.imread("/media/DATA_Old/Face_triton/models/FACE_Triton/5.jpeg")
    image=render_mark(img)
    cv2.imwrite("anh2.jpg",image)
