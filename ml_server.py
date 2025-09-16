import os
import torch
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device



app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

WEIGHTS_PATH = "runs/train/exp9/weights/best.pt"



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
                cv2.putText(img0, label, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)

    return img0

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result_img = detect_objects(filepath, WEIGHTS_PATH)

            result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
            cv2.imwrite(result_path, result_img)

            return render_template("result.html", result_image= result_path)
        
    return render_template("upload.html")


@app.route('/blog')
def blog():
    return render_template('blog.html')  

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == "__main__":
    app.run(debug=True)

