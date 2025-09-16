import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device

def detect_objects(image_path, weights_path, img_size=640, conf_thres=0.5, iou_thres=0.45):

    device = select_device('') 
    model = DetectMultiBackend(weights_path, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    
   
    img0 = cv2.imread(image_path)
    img = cv2.resize(img0, (img_size, img_size))
    img = img.transpose((2, 0, 1))[::-1].copy()  
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  
    if len(img.shape) == 3:
        img = img[None]  

   
    pred = model(img, augment=False, visualize=False)

   
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

   
    for i, det in enumerate(pred):  
        if len(det):
            
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

           
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img0, c1, c2, (255, 0, 0), 2)  
                cv2.putText(img0, label, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img0

if __name__ == '__main__':
   
    input_image_path = '/Users/brian/Downloads/Sorry, but There’s No Such Thing as the “Clean Part” of Moldy Bread.jpg'  
    custom_weights_path = '/Users/brian/BactoScan/yolov5/runs/train/exp9/weights/best.pt'  
    
    result_image = detect_objects(input_image_path, custom_weights_path)
    cv2.imshow('Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
