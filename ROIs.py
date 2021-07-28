# Imports
import cv2 as cv
import os
import numpy as np
import pickle

# Function to rotate the image (this will be used for data augmentation)
def rotate_image(img, angle):
    (h,w)=img.shape[:2]
    rotPoint = (w/2, h/2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (w,h)
    return cv.warpAffine(img, rotMat, dim)

# Configure YOLOv3 and its weights. 
# yolov3.cfg and yolov3.weights can be downloaded here: https://pjreddie.com/darknet/yolo/
model_cfg = "yolov3.cfg"
model_weights = "yolov3.weights"


# Model configuration
net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# It is had a folder containing both lateral and frontal CCS videos
# For every video (lateral and frontal), the ROIs will be obtained
folder = input("Enter the path of the folder which contains the videos: ")

# Function to get the ROIs
# It returns the ROI, as well as four coordinates that will be used for creating the dataset
# These coordinates are x,y,w,h, where w stands for width and h for height
def find_objects(outputs, img):
    hT, wT, channel = img.shape
    bbox = []  #x,y,w,h values
    confs = []
    max_w = 0
    max_h = 0
    for e in outputs:
        for detection in e:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5 and classId == 0: # Only the class person will be detected
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - (w / 2)), int((detection[1] * hT) - h / 2)
                if w*h > max_w*max_h:
                    max_w = w
                    max_h = h
                    bbox = []
                    confs = []
                    bbox.append([x, y, w, h])
                    confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bbox, confs, 0.5, nms_threshold=0.3)
    i=0
    for e in indices:
        e = e[0]
        box = bbox[e]
        x,y,w,h = box[0], box[1], box[2], box[3]
        roi = img[(y - 60):(y + h+60), (x-35):(x + w+35)]
        i+=1
    return roi,x,y,w,h

# Get the ROIs for every video and store them
l=['/Frontal','/Lateral']
d={}
d1={}

d_coord={}
d_coord_aux={}
for e in l: # Frontal, Lateral
    os.chdir(folder + e)
    print(e)
    for video in os.listdir(folder + e): # Video means each athlete
        print(video)
        name=video.split('.')
        name=name[0]
        l_frames=[]
        l_coordinates=[]
        cap = cv.VideoCapture(folder+e+'/'+video) # Open the video
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            l_frames_aux=[]
            l_coordinates_aux=[]
            l_current_frame=[]
            if ret == False:
                break
            l_current_frame.append(frame)

            random_angle = (np.random.uniform() - 0.5) * 20 # The angles for the data augmentation will uniformly be [-10, 10)
            current_rotated_frame = rotate_image(frame, random_angle)
            l_current_frame.append(current_rotated_frame)

            random_angle = (np.random.uniform() - 0.5) * 20
            current_rotated_frame = rotate_image(frame, random_angle)
            l_current_frame.append(current_rotated_frame)

            for frames_and_rotated in l_current_frame:
                blob = cv.dnn.blobFromImage(frames_and_rotated, 1 / 255, (320,320), [0, 0, 0], swapRB=True, crop=False) # A blob is required for YOLOv3
                net.setInput(blob)
                layer_names = net.getLayerNames()
                output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(output_names)
                try: # This Try/Except sentence is just to make sure that the program does not give an error when a ROI is not detected in a frame
                    roi,x,y,w,h = find_objects(outputs, frame)
                    if roi.size != 0:
                        imS = cv.resize(roi, (350, 450))
                        l_frames_aux.append(imS)
                        l_coordinates_aux.append((x,y,w,h))
                except:
                    print('Unable to find a ROI')

            if len(l_frames_aux)==3 and len(l_coordinates_aux)==3: # Store the ROIs and the four coordinates
                l_coordinates.append(l_coordinates_aux)
                l_frames.append(l_frames_aux)


        if len(l_coordinates)>0 and len(l_frames)>0:
            d1[name]=l_frames
            d_coord_aux[name]=l_coordinates
            cap.release()
            cv.destroyAllWindows()

    d[e[1:]]=d1
    d1={}
    d_coord[e[1:]]=d_coord_aux
    d_coord_aux={}

    
# Store the output in pickle files
os.chdir(r'C:\Users\alberto.mira\PycharmProjects\pythonProject')
#ROIs contains the information related to all ROIs found in a Video
ROIs = open('ROIs_.pkl', 'wb')
pickle.dump(d, ROIs)
ROIs.close()


# Coord contains the information related to the coordinates of all ROIs found in a Video
coord = open('Coord_.pkl', 'wb')
pickle.dump(d_coord,coord)
coord.close()




