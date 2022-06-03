from ast import For
from pyparsing import And
import torch
import cv2
import numpy as np

class ObjectCountingAPI:

    def __init__(self, cap, options):
        self.options = options
        self.cap = cap
        self.FRAME_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.FRAME_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.CAP_FPS= cap.get(cv2.CAP_PROP_FPS)
        self.COUNT  = 0

    def get_current_detection_list(self, results, detect_helf_frame) :
        current_detection_list = []
        detection_len = len(results.pandas().xyxy[0])
        for i in range(detection_len):
            xmin = results.pandas().xyxy[0].xmin[i]
            ymin = results.pandas().xyxy[0].ymin[i]
            xmax = results.pandas().xyxy[0].xmax[i]
            ymax = results.pandas().xyxy[0].ymax[i]
            confidence =  results.pandas().xyxy[0].confidence[i]
            class_number =  results.pandas().xyxy[0]['class'][i]
            name =  results.pandas().xyxy[0].name[i]
            detected_object = {
                'xmin' : xmin,
                'ymin' : ymin,
                'xmax' : xmax,
                'ymax' : ymax,
                'confidence' : confidence,
                'class_number' : class_number,
                'name' : name
            }
            if ((detect_helf_frame and  ymin > int(self.FRAME_HEIGHT /2)) or  not detect_helf_frame ) and float(confidence) >=0.50 :
                current_detection_list.append(detected_object)
        
        return  current_detection_list

    def draw_prediction(self, frame, detection):      
        
        label = detection['name']
        color = (0, 0, 255)
        cv2.rectangle(frame, (int(detection['xmin']), 
        int(detection['ymin'])), (int(detection['xmax']), int(detection['ymax'])), color, 2)
        cv2.putText(frame, label, (int(detection['xmin'])-10, int(detection['ymin'])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        




    def count_objects_on_video(self, cap, show=False ,detect_helf_frame =True):
        
        
        
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #  car truck 
        model.classes = [2,7]
        SELECTED_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(SELECTED_DEVICE)
        model.to(device)
        ret, frame = cap.read()
        while ret:

            results = model(frame)
            current_detection_list = []
            current_detection_list = self.get_current_detection_list(results, detect_helf_frame)
            
            if show:
                color = (0, 0, 255)
                COUNTER = "COUNT"+" : "+str(self.COUNT)
                cv2.putText(frame, COUNTER, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if detect_helf_frame :
                    cv2.line(frame, (0, int(self.FRAME_HEIGHT /2)), 
                    ( int(self.FRAME_WIDTH) , int(self.FRAME_HEIGHT /2) ), (0, 0, 255), 1)
                for detection in current_detection_list :
                     self.draw_prediction( frame, detection)
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
        
        