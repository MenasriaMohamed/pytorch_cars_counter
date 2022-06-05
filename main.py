import cv2
from object_counting_api import ObjectCountingAPI

options = {}
VIDEO_PATH = "videos/overpass.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
counter = ObjectCountingAPI(cap, options)
counter.count_objects_on_video(cap, show=True, write=True)
