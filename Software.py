#Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import cv2 as cv
import os
import shutil
import pickle
import json
from scipy.signal._peak_finding import find_peaks
from operator import itemgetter

# Configure YOLO and its weights
model_cfg = "yolov3.cfg"
model_weights = "yolov3.weights"

# Model configuration
net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# It is had a folder containing both lateral and frontal XCS videos
# For every video (lateral and frontal), the ROIs will be obtained


#Inputs
folder = input("Enter the path of the folder which contains the videos: ")
openpose = input("Enter the path of the openpose folder: ")
final_folder = input("Enter the path where you want to create the folder to contain the output patterns: ")


# Create the output folder
os.chdir(final_folder)
try:
    os.mkdir('code_output')
    os.chdir(final_folder+'/code_output')
    os.mkdir('Frontal')
    os.mkdir('Lateral')
except:
    print("Folder already exists")

# Function to get the ROIs (same as in ROIs.py)
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
            if confidence > 0.5 and classId == 0:
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

os.chdir(folder)
d_data={}
d1_data={}
for g in os.listdir(folder):
    for video in os.listdir(folder+'/'+g):
        l1=[]
        l2=[]
        l3=[]
        l4=[]
        l5=[]
        l_all_for_d1=[]
        l_feed_NN=[]
        print(video)
        name = video.split('.')
        name = name[0]
        cap = cv.VideoCapture(folder+'/'+g+'/'+video)
        v=0
        d_frame={}
        d_coordinates={}
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (320, 320), [0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            layer_names = net.getLayerNames()
            output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(output_names)

            try:

                roi, x, y, w, h = find_objects(outputs, frame)
                if roi.size != 0:
                    imS = cv.resize(roi, (325, 475))
                    d_frame[str(v)]=imS  # Store the ROIs obtained
                    d_coordinates[str(v)]=(x,y,w,h) # Store the YOLOv3 coordinates
                    v+=1

            except:
                print('Unable to find a ROI')
                
        os.chdir(openpose+'/images')  # ROIs will be processed by OpenPose
        for k in d_frame:
            cv.imwrite(k+'.jpg', d_frame[k])
        os.chdir(openpose)
        os.system('bin\OpenPoseDemo.exe --image_dir images --render_pose 0 --display 0 --write_json keypoints output/')
        os.chdir(openpose)
        shutil.rmtree(openpose+'/images')
        try:
            os.mkdir('images')
        except:
            print("Folder already exists")
        for j in os.listdir(openpose+'/keypoints'):
            with open(openpose+'/keypoints/'+j) as myfile:
                data=myfile.read()
            obj = json.loads(data)
            obj = obj['people']
            current_image=j.split('_')
            current_image=current_image[0]


            if len(obj) > 1: # Decide which skeleton is the right one
                l = []
                for e in obj:
                    for x in e:
                        if x == 'pose_keypoints_2d':
                            l.append(e[x]) # Get all skeletons read by OpenPose
                j = 0
                # Booleans to detect when a json value is 0
                no_0 = False
                no_1 = False
                no_2 = False
                no_3 = False
                no_4 = False
                no_5 = False
                no_6 = False
                no_7 = False
                no_8 = False
                no_9 = False
                no_10 = False
                no_11 = False
                no_12 = False
                no_13 = False
                no_14 = False
                no_15 = False
                no_16 = False
                no_17 = False
                no_18 = False
                no_19 = False
                no_20 = False
                no_21 = False
                no_22 = False
                no_23 = False
                no_24 = False
                l_total_sum = []  # Store the sum of all the skeletons
                for e in l:
                    l_sum = []  # Store the sum of the current skeleton
                    i = 0
                    sum = 0
                    while i < len(e):
                        if i == 0:
                            if e[i + 1] == 0:
                                no_0 = True
                        elif i == 3:
                            if e[i] == 0:
                                no_1 = True

                        elif i == 6:
                            if e[i + 1] == 0:
                                no_2 = True
                        elif i == 9:
                            if e[i + 1] == 0:
                                no_3 = True
                        elif i == 12:
                            if e[i + 1] == 0:
                                no_4 = True

                        elif i == 15:
                            if e[i + 1] == 0:
                                no_5 = True
                        elif i == 18:
                            if e[i + 1] == 0:
                                no_6 = True
                        elif i == 21:
                            if e[i] == 0:
                                no_7 = True
                        elif i == 24:
                            if e[i] == 0:
                                no_8 = True
                        elif i == 27:
                            if e[i + 1] == 0:
                                no_9 = True
                        elif i == 30:
                            if e[i + 1] == 0:
                                no_10 = True
                        elif i == 33:
                            if e[i + 1] == 0:
                                no_11 = True
                        elif i == 36:
                            if e[i + 1] == 0:
                                no_12 = True
                        elif i == 39:
                            if e[i + 1] == 0:
                                no_13 = True
                        elif i == 42:
                            if e[i + 1] == 0:
                                no_14 = True
                        elif i == 45:
                            if e[i] == 0:
                                no_15 = True
                        elif i == 48:
                            if e[i] == 0:
                                no_16 = True
                        elif i == 51:
                            if e[i] == 0:
                                no_17 = True
                        elif i == 54:
                            if e[i + 1] == 0:
                                no_18 = True
                        elif i == 57:
                            if e[i] == 0:
                                no_19 = True
                        elif i == 60:
                            if e[i] == 0:
                                no_20 = True
                        elif i == 63:
                            if e[i] == 0:
                                no_21 = True
                        elif i == 66:
                            if e[i] == 0:
                                no_22 = True
                        elif i == 69:
                            if e[i] == 0:
                                    no_23 = True
                            if e[i + 3] == 0:
                                no_24 = True
                        i += 3
                    i = 0
                    while i < len(e):
                        if i == 0:
                            if no_0 != True and no_1 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 3:
                            if no_1 != True and no_2 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))
                        elif i == 6:
                            if no_2 != True and no_3 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 9:
                            if no_3 != True and no_4 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 12:
                            if no_4 != True and no_5 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 15:
                            if no_5 != True and no_6 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 18:
                            if no_6 != True and no_7 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 21:
                            if no_7 != True and no_8 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))
                        elif i == 24:
                            if no_8 != True and no_9 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 27:
                            if no_9 != True and no_10 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 30:
                            if no_10 != True and no_11 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 33:
                            if no_11 != True and no_12 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))
                        elif i == 36:
                            if no_12 != True and no_13 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 39:
                            if no_13 != True and no_14 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 42:
                            if no_14 != True and no_15 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 45:
                            if no_15 != True and no_16 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 48:
                            if no_16 != True and no_17 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 51:
                            if no_17 != True and no_18 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 54:
                            if no_18 != True and no_19 != True:
                                l_sum.append(abs(e[i + 1] - e[i + 4]))

                        elif i == 57:
                            if no_19 != True and no_20 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 60:
                            if no_20 != True and no_21 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 63:
                            if no_21 != True and no_22 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))

                        elif i == 66:
                            if no_22 != True and no_23 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))
                        elif i == 69:
                            if no_23 != True and no_24 != True:
                                l_sum.append(abs(e[i] - e[i + 3]))
                        i += 3
                    a = 0.0
                    p = 0
                    while p < len(l_sum):
                        a = a + l_sum[p]
                        p += 1
                    if a > sum:
                        if len(l_total_sum) == 0:
                            l_total_sum.append(a)
                            l_total_sum.append(j)
                        else:
                            if a > l_total_sum[0]:
                                l_total_sum[0] = a
                                l_total_sum[1] = j
                    j += 1
                d = obj[l_total_sum[1]]
                l_points = d['pose_keypoints_2d']
                i = 0
                arr = np.zeros((1, 2))
                while i < len(l_points):
                    if len(arr) == 1 and arr[0][0]==0 and arr[0][1]==0:
                        arr[0][0] = l_points[i]
                        arr[0][1] = l_points[i + 1]
                    else:
                        arr = np.append(arr, [[l_points[i], l_points[i + 1]]], axis=0)
                    i = i + 3
                coord_2 = d_coordinates[current_image]
                arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
            elif len(obj)==1:
                d = obj[0]
                l = d['pose_keypoints_2d']
                i = 0
                arr = np.zeros((1, 2))
                while i < len(l):
                    if len(arr) == 1 and arr[0][0] == 0 and arr[0][1] == 0:
                        arr[0][0] = l[i]
                        arr[0][1] = l[i + 1]
                    else:
                        arr = np.append(arr, [[l[i], l[i + 1]]], axis=0)
                    i += 3
                coord_2 = d_coordinates[current_image]
                arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
            if len(obj)>=1:
                l_feed_NN.append((arr, current_image))

        os.chdir(openpose)
        shutil.rmtree(openpose+'/keypoints')
        try:
            os.mkdir('keypoints')
        except:
            print("Folder already exists")
        l_all_for_d1.append(l_feed_NN) # Numpy arrays and the frame they belong in
        l_all_for_d1.append(d_frame) # ROIs and their frame number
        d1_data[name]=l_all_for_d1
    d_data[g]=d1_data
    d1_data={} # Store the data in a dictionary


# Processing the data with the Neural Networks. TensorFlow version: 2.5.0
for e in d_data: #Lateral,  frontal
    a=d_data[e]
    for j in a:
        b=a[j] # (27, i) (ROI, i)
        os.chdir(r'C:\Users\alberto.mira\PycharmProjects\pythonProject')
        if e == 'Lateral':
            model = tf.keras.models.load_model('Lateral_model.h5')
        elif e == 'Frontal':
            model = tf.keras.models.load_model('Frontal_model.h5')
        l1,l2,l3,l4,l5=[],[],[],[],[]
        n_arrays=b[0]
        l_test = []
        for k in n_arrays:
            if k[0].shape == (27, 2):
                a = tf.convert_to_tensor(k[0])
                a = tf.reshape(a, (6, 9, 1))
                l_test.append(a)
                a = np.array(l_test)
                a = model.predict(a)
                a = tf.nn.softmax(a)
                l_test = []
                a1 = a[0][0]
                a2 = a[0][1]
                a3 = a[0][2]
                a4 = a[0][3]
                a5 = a[0][4]
                l1.append((float(a1), k[1]))  # These lists contain all the probabilities of a single athlete in either a lateral or a frontal video
                l2.append((float(a2), k[1]))
                l3.append((float(a3), k[1]))
                l4.append((float(a4), k[1]))
                l5.append((float(a5), k[1]))

        probs_1 = [item[0] for item in l1]
        probs_2 = [item[0] for item in l2]
        probs_3 = [item[0] for item in l3]
        probs_4 = [item[0] for item in l4]
        probs_5 = [item[0] for item in l5]




        fr_1 = [item[1] for item in l1]
        fr_2 = [item[1] for item in l2]
        fr_3 = [item[1] for item in l3]
        fr_4 = [item[1] for item in l4]
        fr_5 = [item[1] for item in l5]


        # Seek the cycles by setting the prominence. If no cycle is found, the prominence will go on decreasing 0.05 points until the algorithm finally finds at least 
        # a cycle where all the five patterns overcome that prominence
        promin = 0.9
        have_cycle = False
        l_sequences = []
        while promin > 0 and have_cycle==False:
            peaks_1 = find_peaks(probs_1, prominence=promin)
            peaks_2 = find_peaks(probs_2, prominence=promin)
            peaks_3 = find_peaks(probs_3, prominence=promin)
            peaks_4 = find_peaks(probs_4, prominence=promin)
            peaks_5 = find_peaks(probs_5, prominence=promin)


            l_peaks = []
            if len(peaks_1[0]) > 0 and len(peaks_2[0]) > 0 and len(peaks_3[0]) > 0 and len(peaks_4[0]) > 0 and len(peaks_5[0]) > 0:
                peak = 0
                while peak < len(peaks_1[0]):
                    current_peak = peaks_1[0][peak]
                    l_peaks.append((int(fr_1[current_peak]), probs_1[current_peak], 1))
                    peak += 1
                peak = 0
                while peak < len(peaks_2[0]):
                    current_peak = peaks_2[0][peak]
                    l_peaks.append((int(fr_2[current_peak]), probs_2[current_peak], 2))
                    peak += 1
                peak = 0
                while peak < len(peaks_3[0]):
                    current_peak = peaks_3[0][peak]
                    l_peaks.append((int(fr_3[current_peak]), probs_3[current_peak], 3))
                    peak += 1
                peak = 0
                while peak < len(peaks_4[0]):
                    current_peak = peaks_4[0][peak]
                    l_peaks.append((int(fr_4[current_peak]), probs_4[current_peak], 4))
                    peak += 1
                peak = 0
                while peak < len(peaks_5[0]):
                    current_peak = peaks_5[0][peak]
                    l_peaks.append((int(fr_5[current_peak]), probs_5[current_peak], 5))
                    peak += 1

            l_peaks = sorted(l_peaks, key=itemgetter(0))


            counter1 = 0
            counter2 = 0
            counter3 = 0
            counter4 = 0
            counter5 = 0

            frame1 = 0
            frame2 = 0
            frame3 = 0
            frame4 = 0
            frame5 = 0

            prob1 = 0
            prob2 = 0
            prob3 = 0
            prob4 = 0
            prob5 = 0

            l_sequences_aux = []


            for find_patterns in l_peaks:

                if counter1 == 1 and counter2 == 1 and counter3 == 1 and counter4 == 1 and counter5 == 1:
                    have_cycle = True
                    l_sequences_aux.append((frame1, frame2, frame3, frame4, frame5))
                    l_sequences_aux.append((prob1, prob2, prob3, prob4, prob5))
                    l_sequences.append(l_sequences_aux)
                    l_sequences_aux = []
                    counter1,counter2,counter3,counter4,counter5=0,0,0,0,0

                else:
                    if find_patterns[2] == 1:
                        counter1 = 1
                        frame1 = find_patterns[0]
                        prob1 = find_patterns[1]


                    if find_patterns[2] == 2 and counter1 == 1:
                        if counter2 == 0:
                            counter2 = 1
                            frame2 = find_patterns[0]
                            prob2 = find_patterns[1]
                        else:
                            if find_patterns[1] > prob2:
                                frame2 = find_patterns[0]
                                prob2 = find_patterns[1]

                    if find_patterns[2] == 3 and counter2 == 1 and counter1 == 1:
                        if counter3 == 0:
                            counter3 = 1
                            frame3 = find_patterns[0]
                            prob3 = find_patterns[1]
                        else:
                            if find_patterns[1]>prob3:
                                frame3 = find_patterns[0]
                                prob3 = find_patterns[1]


                    elif find_patterns[2] == 3 and (counter2 == 0 or counter1 == 0):
                        counter1 = 0
                        counter2 = 0


                    if find_patterns[2] == 4 and counter3 == 1 and counter2 == 1 and counter1 == 1:
                        if counter4 == 0:
                            counter4 = 1
                            frame4 = find_patterns[0]
                            prob4 = find_patterns[1]
                        else:
                            if find_patterns[1]>prob4:
                                frame4 = find_patterns[0]
                                prob4 = find_patterns[1]

                    elif find_patterns[2] == 4 and (counter3 == 0 or counter2 == 0 or counter1 == 0):
                        counter1 = 0
                        counter2 = 0
                        counter3 = 0

                    if find_patterns[2] == 5 and counter4 == 1 and counter3 == 1 and counter2 == 1 and counter1 == 1:
                        if counter5 == 0:
                            counter5 = 1
                            frame5 = find_patterns[0]
                            prob5 = find_patterns[1]
                        else:
                            if find_patterns[1]>prob5:
                                frame5 = find_patterns[0]
                                prob5 = find_patterns[1]
                    elif find_patterns[2] == 5 and (counter4 == 0 or counter3 == 0 or counter2 == 0 or counter1 == 0):
                        counter4 = 0
                        counter3 = 0
                        counter2 = 0
                        counter1 = 0
            
            promin = promin - 0.05

        last_output = 0
        
        
        # Get the ROIs
        b_ROIs= b[1]
        while last_output < len(l_sequences): # Go through the sequences detected and their probabilities (sequence = cycle)
            os.chdir(final_folder +'/code_output/'+ e)
            try:
                os.mkdir(name)
            except:
                print("Folder already exists")


            os.chdir(final_folder+'/code_output/' + e + '/' + name)

            extension = len(os.listdir(final_folder + '/code_output/' + e + '/' + name))
            try:
                os.mkdir(str(extension+1))
            except:
                print("Folder already exists")

            os.chdir(final_folder + '/code_output/' + e + '/' + name + '/' + str(extension+1))

            last_output_2=l_sequences[last_output]
            cv.imwrite('Pattern_1.jpg', b_ROIs[str(last_output_2[0][0])])  # Write the ROIs into the folder
            cv.imwrite('Pattern_2.jpg', b_ROIs[str(last_output_2[0][1])])
            cv.imwrite('Pattern_3.jpg', b_ROIs[str(last_output_2[0][2])])
            cv.imwrite('Pattern_4.jpg', b_ROIs[str(last_output_2[0][3])])
            cv.imwrite('Pattern_5.jpg', b_ROIs[str(last_output_2[0][4])])

            f = open("Probabilities.txt", "w") # Generate a .txt file where the probability of each pattern appears
            f.write("Probability 1: " + str(last_output_2[1][0]) + '\n')
            f.write("Probability 2: " + str(last_output_2[1][1]) + '\n')
            f.write("Probability 3: " + str(last_output_2[1][2]) + '\n')
            f.write("Probability 4: " + str(last_output_2[1][3]) + '\n')
            f.write("Probability 5: " + str(last_output_2[1][4]) + '\n')
            f.close()
            last_output += 1

        l_sequences = []
