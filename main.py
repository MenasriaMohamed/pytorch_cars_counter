import cv2
from object_counting_api import ObjectCountingAPI

options = {}
VIDEO_PATH = "videos/overpass.mp4"


cap = cv2.VideoCapture(VIDEO_PATH)
counter = ObjectCountingAPI(cap,options)

counter.count_objects_on_video(cap, show=True , detect_helf_frame =True)

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# # model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024,len(['person']))
# # Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# # Inference
# results = model(imgs)

# # Results
# # results.print()
# # results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)

# print(results.pandas().xyxy[0].name[0])
# print(results.pandas().xyxy[0].name[1])
# print(results.pandas().xyxy[0].name[2])
# print(results.pandas().xyxy[0].name[3])

