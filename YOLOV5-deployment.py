#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
from tqdm.auto import tqdm
import shutil as sh

from PIL import Image, ImageDraw

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import cv2
import torch

YOLO_PATH = "/home/mohamed-ali/Downloads/yolov5/runs/train/cheque_yolov5s8/weights/best.pt"


# In[39]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights_path = {"YOLO":YOLO_PATH}


# ### Torch hub deployment

# In[40]:


class ChequeDetection:
    
    """
    Class implements Yolo5 model to make inferences on a banque cheque.
    """
    
    def __init__(self, image_url, out_file, weights_path):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as URL image,on which prediction is made.
        :param out_file: A valid output file name.
        :param weights_path: Dict of models weight paths
        """
        self._URL = image_url
        #self.model = self.load_model(model_name=) # Load model weights
        #self.classes = self.model.names # Classes name
        self.out_file = out_file # Output file directory
        self.weights_path = weights_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Device 
        print("\n\nDevice Used:",self.device)
        
    def get_image_from_url(self):
        """
        Creates a new image object to extract fields from it using yolov5.
        :return: opencv2 image.
        """
        
        assert self.image_url is not None
        return cv2.imread(self.image_url)
    
    def load_model(self, model_name):
        """
        Loads DNN model from weights path.
        :return: Trained Pytorch model.
        """
        
        if self.weights_path[model_name] == "YOLO":
            model = torch.hub.load('ultralytics/yolov5', "custom", path="/home/mohamed-ali/Downloads/yolov5/runs/train/essaie12/weights/best.pt", force_reload=True)
        return model
    
    
    def score_frame(self, img):
        """
        Takes a single img as input, and scores the img using yolo5 model.
        :param img: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(img)
     
        confidence, cord = results.xyxyn[0][:, -3], results.xyxyn[0][:, :-1]
        return confidence, cord
    
    
    def draw_boxes(img, results):
        """
        Takes an image and its results as input, and plots the bounding boxes and label on to the image.
        :param results: contains labels and coordinates predicted by model on the given img.
        :param img: img which has been scored.
        :return: img with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = img.shape[1], img.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 1)
                torch.set_printoptions(precision=3)
                cv2.putText(img, str(labels[i].item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1)
        
        return img


# In[41]:


img_url = "/home/mohamed-ali/Downloads/yolov5/cheque_data/validation/1.jpg"
out_dir = "/home/mohamed-ali/Downloads/output_files"


# In[37]:


cheque_detection = ChequeDetection(img_url, out_dir, weights_path)


# In[42]:


#def draw_box(img, boxes):
#    """
#    Draw boxes on the picture
#    :param img : PIL Image object
#    :param boxes : numpy array of size [number_of_boxes, 6]
#    :return : PIL Image object with rectangles
#    """
#    box = ImageDraw.Draw(img)
#    for i in range(boxes.shape[0]):
#        data = list(boxes[i])
#        shape = [data[0], data[1], data[2], data[3]]
#        box.rectangle(shape, outline ="#02d5fa", width=3)
#    return img

def draw_box(img, boxes):
    """
    Draw boxes on the picture
    :param img : PIL Image object
    :param boxes : numpy array of size [number_of_boxes, 6]
    :return : PIL Image object with rectangles
    """
    box = ImageDraw.Draw(img)
    for i in range(boxes.shape[0]):
        data = list(boxes[i].cpu().numpy()) # Shut gpu for numpy usage
        shape = [data[0], data[1], data[2], data[3]]
        box.rectangle(shape, outline ="#02d5fa", width=3)
    return img

def plot_boxes(img, results):
        """
        Takes an image and its results as input, and plots the bounding boxes and label on to the image.
        :param results: contains labels and coordinates predicted by model on the given img.
        :param img: img which has been scored.
        :return: img with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = img.shape[1], img.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 1)
                torch.set_printoptions(precision=3)
                cv2.putText(img, str(labels[i].item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1)
        
        return img
                
def class_to_label(classes, x):
    return classes[int(x)]


# In[43]:


img_cv2 = cv2.imread("/home/mohamed-ali/Downloads/yolov5/cheque_data/validation/1.jpg")
img_pillow = Image.open("/home/mohamed-ali/Downloads/yolov5/cheque_data/train/2001.jpg")
model2 = torch.hub.load('ultralytics/yolov5', "custom", path="/home/mohamed-ali/Downloads/yolov5/runs/train/essaie12/weights/best.pt", force_reload=True)


# In[44]:


cv2_results = model2(img_cv2, size=512, augment=True)


# In[45]:


pil_results = model2(img_pillow, size=512, augment=True)


# In[46]:


pil_results.pandas().xyxy[0]


# In[47]:


cv2_results.pandas().xyxy[0]


# In[ ]:


crops = cv2_results.crop(save=True, save_dir=out_file)
#cv2_results.crop()


# In[ ]:


cv2_results.crop


# In[ ]:


results = cv2_results.xyxyn[0][:, -3], cv2_results.xyxyn[0][:, :-1]


# In[8]:


pil_boxes = pil_results.xyxyn[0][:, -3], pil_results.xyxyn[0][:, :-1]


# In[16]:


_, cord = pil_boxes


# In[27]:


img_pillow = draw_box(img_pillow, cord)


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(30, 70))
ax.imshow(draw_pil)


# In[11]:


data = list(cord[0].cpu().numpy())


# In[12]:


data


# In[ ]:


cord.shape[0]


# In[ ]:


draw_img = draw_box(img_pillow, data)


# In[ ]:


box = ImageDraw.Draw(img_pillow)


# In[ ]:


cord


# In[ ]:


results


# In[ ]:


test = plot_boxes(img_cv2, results)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(30, 70))
ax.imshow(test)


# In[ ]:


frame_boxes = plot_boxes(results, img_cv2)


# In[ ]:


labels, cord = results


# In[ ]:


labels


# In[ ]:


cord


# In[ ]:


labels, cord = results
n = len(labels)
x_shape, y_shape = frame.shape[1], frame.shape[0]
for i in range(n):
    row = cord[i]
    if row[4] >= 0.2:
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(30, 70))
classes = model2.names
labels, cord = results
n = len(labels)
img_width, img_height = img_cv2.shape[1], img_cv2.shape[0]
for i in range(n):
    row = cord[i]
    if row[4] >= 0.2:
        x, y, h, w = int(row[0]*img_width), int(row[1]*img_height), int(row[2]*img_width), int(row[3]*img_height)
        bgr = (255, 0, 0)
        cv2.rectangle(img_cv2, (x, y), (h, w), bgr, 1)
        cv2.putText(img_cv2, class_to_label(classes, labels[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
ax.imshow(img_cv2)


# In[ ]:


x,y,h,w = int(x*img_width), int(y*img_height), int(h*img_higth), int(w*img_width)


# In[ ]:


row


# In[ ]:


x, y, h, w = int(row[0]*img_width), int(row[1]*img_height), int(row[2]*img_height), int(row[3]*img_width)


# In[ ]:


class_to_label(classes, labels[i])


# In[ ]:


int(labels[1])


# In[ ]:


plt.imshow(img_cv2)


# In[ ]:


labels, cord = results


# In[ ]:


labels


# In[ ]:


classes = model2.names


# In[ ]:




