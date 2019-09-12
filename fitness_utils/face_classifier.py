import cv2
import os
def get_face_fitness(image):
    cascade = cv2.CascadeClassifier()
    model_path = os.path.join(os.getcwd(), 'fitness_utils/model.xml')
    cascade.load(model_path)
    faces, lvls , ws = cascade.detectMultiScale3(image,1.2, 3, outputRejectLevels=True)
    if lvls == ():
        return -1
    else:
        return lvls[0]


"""
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
  
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
  
    # return the intersection over union value
    return iou

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(image,(435,41),(435+345,41+485),(255,0,0),2)
 
gt = [435,41,345,485]
 
face_dt = [faces[0,0],faces[0,1],faces[0,0]+faces[0,2],faces[0,1]+faces[0,3]]
gt_dt = [gt[0],gt[1],gt[0]+gt[2], gt[1]+gt[3]]
 
print(bb_intersection_over_union(face_dt ,gt_dt))
 
cv2.imshow('image detections',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""