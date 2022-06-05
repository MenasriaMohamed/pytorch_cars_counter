from ast import For
from pyparsing import And
import torch
import cv2
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon

GLOBAL_COLOR = (0, 0, 255)


class ObjectCountingAPI:

    def __init__(self, cap, options):
        self.options = options
        self.cap = cap
        self.FRAME_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.FRAME_HEIGHT = cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.CAP_FPS = cap.get(cv2.CAP_PROP_FPS)
        self.COUNT = 0
        self.detection_list = []
        # //////////////////
        self.LINE_X_START = +0
        self.LINE_Y_START = int(self.FRAME_HEIGHT / 2 + 70)
        self.LINE_X_END = int(self.FRAME_WIDTH - 0)
        self.LINE_Y_END = int(self.FRAME_HEIGHT / 2 + 70)

    def get_distance(self, center_x_1, center_y_1, center_x_2, center_y_2):
        p = [center_x_1, center_y_1]
        q = [center_x_2, center_y_2]
        # Calculate Euclidean distance
        return math.dist(p, q)

    def intersection_old_new(self, detected_object, nearest):
        # intersection = self.get_intersection_with_rectangles(
        #     detected_object, nearest)
        intersection = self.get_intersection_point_rectangle(
            detected_object, nearest['center_x'], nearest['center_y'])
        return intersection

    def remove_all_nearest(self, detected_object):
        new_list = []
        for detection in self.detection_list:
            intersection = self.intersection_old_new(
                detected_object, detection)
            if(not intersection):
                new_list.append(detection)

        self.detection_list = new_list

    def get_nearest_detection(self, detected_object):
        nearest = None
        min_distance = self.FRAME_WIDTH + self.FRAME_HEIGHT + 1
        for detection in self.detection_list:
            distance = self.get_distance(
                detection['center_x'], detection['center_y'], detected_object['center_x'], detected_object['center_y'])
            if(distance < min_distance):
                min_distance = distance
                nearest = detection

        if(nearest != None):
            intersection = self.intersection_old_new(
                detected_object, nearest)
            if not intersection:
                nearest = None
        return nearest

    def get_intersection_with_rectangles(self, detected_object, nearest):
        line1 = LineString([(detected_object['xmin'], detected_object['ymin']),
                            (detected_object['xmax'], detected_object['ymin']),
                            (detected_object['xmax'], detected_object['ymax']),
                            (detected_object['xmin'], detected_object['ymax'])
                            ])
        line2 = LineString([(nearest['xmin'], nearest['ymin']),
                            (nearest['xmax'], nearest['ymin']),
                            (nearest['xmax'], nearest['ymax']),
                            (nearest['xmin'], nearest['ymax'])
                            ])
        if(line1.intersection(line2)):
            return True
        #  LINESTRING EMPTY
        return False

    def get_intersection_point_rectangle(self, detected_object, xp, yp):
        p1 = Point(xp, yp)
        coords = [(detected_object['xmin'], detected_object['ymin']),
                  (detected_object['xmax'], detected_object['ymin']),
                  (detected_object['xmax'], detected_object['ymax']),
                  (detected_object['xmin'], detected_object['ymax'])
                  ]
        poly = Polygon(coords)

        return p1.within(poly)

    def get_intersection_with_line(self, detected_object):
        # line1 = LineString([(detected_object['xmin'], detected_object['ymin']),
        #                     (detected_object['xmax'], detected_object['ymin']),
        #                     (detected_object['xmax'], detected_object['ymax']),
        #                     (detected_object['xmin'], detected_object['ymax'])
        #                     ])
        coords = [(detected_object['xmin'], detected_object['ymin']),
                  (detected_object['xmax'], detected_object['ymin']),
                  (detected_object['xmax'], detected_object['ymax']),
                  (detected_object['xmin'], detected_object['ymax'])
                  ]
        poly = Polygon(coords)
        line = LineString(
            [(self.LINE_X_START, self.LINE_Y_START), (self.LINE_X_END, self.LINE_Y_END)])
        if(line.intersection(poly)):
            return True
        #  LINESTRING EMPTY
        return False

    def get_current_detection_list(self, results):
        current_detection_list = []
        detection_len = len(results.pandas().xyxy[0])
        for i in range(detection_len):
            xmin = results.pandas().xyxy[0].xmin[i]
            ymin = results.pandas().xyxy[0].ymin[i]
            xmax = results.pandas().xyxy[0].xmax[i]
            ymax = results.pandas().xyxy[0].ymax[i]
            center_x = ((xmax + xmin) / 2)
            center_y = ((ymax + ymin) / 2)
            confidence = results.pandas().xyxy[0].confidence[i]
            class_number = results.pandas().xyxy[0]['class'][i]
            name = results.pandas().xyxy[0].name[i]
            detected_object = {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'confidence': confidence,
                'class_number': class_number,
                'name': name,
                'id': -1,
                'center_x':  center_x,
                'center_y': center_y
            }
            intersection_with_line = self.get_intersection_with_line(
                detected_object)
            if not intersection_with_line:
                self.remove_all_nearest(detected_object)

            if float(confidence) >= 0.50 and intersection_with_line:
                nearest = None
                nearest = self.get_nearest_detection(detected_object)

                if nearest is None:
                    self.COUNT = self.COUNT + 1
                    detected_object['id'] = self.COUNT
                    self.detection_list.append(detected_object)
                else:
                    nearest_id = nearest['id']
                    detected_object['id'] = nearest_id
                    self.detection_list.remove(nearest)
                    self.detection_list.append(detected_object)

                # ////////////////////::

                current_detection_list.append(detected_object)

        return current_detection_list

    def draw_prediction(self, frame, detection):

        label = detection['name'] + " " + str(detection['id'])
        color = (0, 0, 255)
        cv2.rectangle(frame, (int(detection['xmin']),
                              int(detection['ymin'])), (int(detection['xmax']), int(detection['ymax'])), color, 2)

        cv2.circle(frame, (int(detection['center_x']),
                   int(detection['center_y'])), 5, (0, 0, 255), -1)

        cv2.putText(frame, label, (int(detection['xmin'])-10, int(detection['ymin'])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def count_objects_on_video(self, cap, show=False ,write = False):
        if write :
            video_file = 'race-01.mp4'
            image_size = (int(self.FRAME_WIDTH),  int(self.FRAME_HEIGHT))
            fps = self.CAP_FPS
            out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
        

        # Model
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
        #  car truck
        model.classes = [2, 7]
        SELECTED_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(SELECTED_DEVICE)
        model.to(device)
        ret, frame = cap.read()
        while ret:
            results = model(frame)
            current_detection_list = []
            current_detection_list = self.get_current_detection_list(results)

            cv2.putText(frame, "COUNT"+" : "+str(self.COUNT), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, GLOBAL_COLOR, 2)

            cv2.line(frame, (self.LINE_X_START, self.LINE_Y_START),
                        (self.LINE_X_END, self.LINE_Y_END), GLOBAL_COLOR, 1)

            for detection in current_detection_list:
                self.draw_prediction(frame, detection)
            
            if write : 
                out.write(frame)

            if show:
                cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
