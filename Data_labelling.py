#Imports
import os
import cv2 as cv
import pickle
import numpy as np
import random

#SIFT
sift= cv.SIFT_create()
# BF
bf = cv.BFMatcher()

# Open the file containing the information related to the already-given patterns
pkl_file = open('Skating_1_1_given_patterns.pkl', 'rb')
patterns= pickle.load(pkl_file)
pkl_file.close()

# Open the file containing the ROIs detected from the videos
pkl_file_2 = open('ROIs_.pkl', 'rb')
ROIs= pickle.load(pkl_file_2)
pkl_file_2.close()

# Open the file containing the four coordinates that YOLOv3 can detect when processing the frames
pkl_file_2 = open('Coord_.pkl', 'rb')
coord= pickle.load(pkl_file_2)
pkl_file_2.close() 

d1={}
d={}
for e in patterns:
    patterns_2=patterns[e]   #Lateral, frontal
    if e=='lateral':
        e="Lateral"
    ROIs_2=ROIs[e]
    coord_2=coord[e]
    print(e)
    for j in patterns_2:   # Name
        l_all_patterns = []
        l_first_pattern = []
        l_second_pattern = []
        l_third_pattern = []
        l_fourth_pattern = []
        l_fifth_pattern = []
        if (j in patterns_2) and (j in ROIs_2) and (j in coord_2): 
            print(j)
            l_frames=patterns_2[j] #Frames related to that name
            l_ROIs=ROIs_2[j]
            l_coord=coord_2[j]
            i=1
            for p in l_frames: # Go through the already-given patterns
                try:
                    im1=cv.cvtColor(p,cv.COLOR_BGR2GRAY)
                    kp1, des1 = sift.detectAndCompute(im1, None)
                except:
                    print('Unable to open the image')

                l_probs = []
                for n in l_ROIs: # Make the comparison between the already-given patterns and all the ROIs detected in a video
                    good = []
                    try:
                        im2=cv.cvtColor(n[0], cv.COLOR_BGR2GRAY)
                        kp2, des2 = sift.detectAndCompute(im2, None)
                        matches = bf.knnMatch(des1, des2, k=2)
                        for m, c in matches:
                            if m.distance < 0.75 * c.distance:
                                good.append([m])
                        a = len(good)
                        l_probs.append(a)
                    except:
                        l_probs.append(0)
                if i==1:
                    for pattern_probs in l_probs:
                        l_first_pattern.append(pattern_probs)
                elif i==2:
                    for pattern_probs in l_probs:
                        l_second_pattern.append(pattern_probs)
                elif i==3:
                    for pattern_probs in l_probs:
                        l_third_pattern.append(pattern_probs)
                elif i==4:
                    for pattern_probs in l_probs:
                        l_fourth_pattern.append(pattern_probs)
                elif i==5:
                    for pattern_probs in l_probs:
                        l_fifth_pattern.append(pattern_probs)
                i+=1
            l_all_patterns.append(l_first_pattern)
            l_all_patterns.append(l_second_pattern)
            l_all_patterns.append(l_third_pattern)
            l_all_patterns.append(l_fourth_pattern)
            l_all_patterns.append(l_fifth_pattern)
            d1[j]=l_all_patterns
    d[e]=d1
    d1={}


# SIFT patterns contains a dictionary like that:
# Lateral and Frontal: Athlete: 5 lists (each list contains the matches with regard to each one of
# the already-given patterns).
os.chdir(r'C:\Users\alberto.mira\PycharmProjects\pythonProject')
dataset = open('SIFT_patterns.pkl', 'wb')
pickle.dump(d, dataset)
dataset.close()


# Open the SIFT_patterns again
pkl_file = open('SIFT_patterns.pkl', 'rb')
dataset= pickle.load(pkl_file)
pkl_file.close()


d={}
d1={}
d2={}
for e in dataset: #e can be either Lateral or Frontal
    a=dataset[e]
    for j in a: #j is the name of the athlete
        pattern_1=0
        pattern_2=0
        pattern_3=0
        pattern_4=0
        pattern_5=0
        l_pattern_6=[]
        b=a[j]
        i=1
        list_length=[]
        for n in b: #n is a list containing the number of matches for each frame
            current_max=max(n)
            ROI_max=n.index(current_max)
            l_pattern_6.append(ROI_max)
            list_length=len(n)

            if i==1:
                pattern_1=ROI_max
            elif i==2:
                pattern_2=ROI_max
            elif i==3:
                pattern_3=ROI_max
            elif i==4:
                pattern_4=ROI_max
            elif i==5:
                pattern_5=ROI_max
            i+=1

        random_number= random.randint(0,(list_length-1))
        i=0
        state=False
        while i<100 and state==False:
            if random_number not in l_pattern_6:
                 state=True
            else:
                random_number = random.randint(0, (list_length-1))
            i+=1

        #Now, it is had all the 5 patterns, which are suposed to be the same as the already-given ones
        #plus the pattern 6

        d2['1']=pattern_1
        d2['2']=pattern_2
        d2['3']=pattern_3
        d2['4']=pattern_4
        d2['5']=pattern_5
        d2['6']=random_number
        d1[j]=d2
        d2={}
    d[e]=d1
    d1={}

    
# Dataset_indexes contains the number of the frame that has been classified as a pattern (it does not contain the ROI directly).
dataset = open('Dataset_indexes.pkl', 'wb')
pickle.dump(d, dataset)
dataset.close()


# Open Dataset_indexes again
pkl_file = open('Dataset_indexes.pkl', 'rb')
final_dataset= pickle.load(pkl_file)
pkl_file.close()

# Finally, two pickle files will be created. One will contain the ROIs which are the final dataset, and the other one, will contain the coordinates.
d2_ROIs={}
d2_coords={}
d1_ROIs={}
d1_coords={}
d_ROIs={}
d_coords={}
for e in final_dataset:
    a = final_dataset[e]
    all_ROIs=ROIs[e]
    all_coords=coord[e]
    for j in a:
        l_pattern_1_ROIs = []
        l_pattern_2_ROIs = []
        l_pattern_3_ROIs = []
        l_pattern_4_ROIs = []
        l_pattern_5_ROIs = []
        l_pattern_6_ROIs = []

        l_pattern_1_coords = []
        l_pattern_2_coords = []
        l_pattern_3_coords = []
        l_pattern_4_coords = []
        l_pattern_5_coords = []
        l_pattern_6_coords = []
        b = a[j]
        if (j in all_ROIs) and (j in all_coords):
            athlete_ROIs= all_ROIs[j]
            athlete_coords=all_coords[j]
            for k in b:
                current_index=b[k]
                if k=='1':
                    current_ROIs_now=athlete_ROIs[current_index]
                    l_pattern_1_ROIs.append(current_ROIs_now[0])
                    l_pattern_1_ROIs.append(current_ROIs_now[1])
                    l_pattern_1_ROIs.append(current_ROIs_now[2])

                    current_coords_now=athlete_coords[current_index]
                    l_pattern_1_coords.append(current_coords_now[0])
                    l_pattern_1_coords.append(current_coords_now[1])
                    l_pattern_1_coords.append(current_coords_now[2])
                elif k=='2':
                    current_ROIs_now = athlete_ROIs[current_index]
                    l_pattern_2_ROIs.append(current_ROIs_now[0])
                    l_pattern_2_ROIs.append(current_ROIs_now[1])
                    l_pattern_2_ROIs.append(current_ROIs_now[2])

                    current_coords_now = athlete_coords[current_index]
                    l_pattern_2_coords.append(current_coords_now[0])
                    l_pattern_2_coords.append(current_coords_now[1])
                    l_pattern_2_coords.append(current_coords_now[2])

                elif k=='3':
                    current_ROIs_now = athlete_ROIs[current_index]
                    l_pattern_3_ROIs.append(current_ROIs_now[0])
                    l_pattern_3_ROIs.append(current_ROIs_now[1])
                    l_pattern_3_ROIs.append(current_ROIs_now[2])

                    current_coords_now = athlete_coords[current_index]
                    l_pattern_3_coords.append(current_coords_now[0])
                    l_pattern_3_coords.append(current_coords_now[1])
                    l_pattern_3_coords.append(current_coords_now[2])

                elif k=='4':
                    current_ROIs_now = athlete_ROIs[current_index]
                    l_pattern_4_ROIs.append(current_ROIs_now[0])
                    l_pattern_4_ROIs.append(current_ROIs_now[1])
                    l_pattern_4_ROIs.append(current_ROIs_now[2])

                    current_coords_now = athlete_coords[current_index]
                    l_pattern_4_coords.append(current_coords_now[0])
                    l_pattern_4_coords.append(current_coords_now[1])
                    l_pattern_4_coords.append(current_coords_now[2])

                elif k=='5':
                    current_ROIs_now = athlete_ROIs[current_index]
                    l_pattern_5_ROIs.append(current_ROIs_now[0])
                    l_pattern_5_ROIs.append(current_ROIs_now[1])
                    l_pattern_5_ROIs.append(current_ROIs_now[2])

                    current_coords_now = athlete_coords[current_index]
                    l_pattern_5_coords.append(current_coords_now[0])
                    l_pattern_5_coords.append(current_coords_now[1])
                    l_pattern_5_coords.append(current_coords_now[2])

                elif k=='6':
                    current_ROIs_now = athlete_ROIs[current_index]
                    l_pattern_6_ROIs.append(current_ROIs_now[0])
                    l_pattern_6_ROIs.append(current_ROIs_now[1])
                    l_pattern_6_ROIs.append(current_ROIs_now[2])

                    current_coords_now = athlete_coords[current_index]
                    l_pattern_6_coords.append(current_coords_now[0])
                    l_pattern_6_coords.append(current_coords_now[1])
                    l_pattern_6_coords.append(current_coords_now[2])

        d2_ROIs['1']=l_pattern_1_ROIs
        d2_ROIs['2']=l_pattern_2_ROIs
        d2_ROIs['3'] = l_pattern_3_ROIs
        d2_ROIs['4'] = l_pattern_4_ROIs
        d2_ROIs['5'] = l_pattern_5_ROIs
        d2_ROIs['6'] = l_pattern_6_ROIs

        d1_ROIs[j]=d2_ROIs
        d2_ROIs={}

        d2_coords['1']=l_pattern_1_coords
        d2_coords['2'] = l_pattern_2_coords
        d2_coords['3'] = l_pattern_3_coords
        d2_coords['4'] = l_pattern_4_coords
        d2_coords['5'] = l_pattern_5_coords
        d2_coords['6'] = l_pattern_6_coords
        d1_coords[j]=d2_coords
        d2_coords={}

    d_ROIs[e]=d1_ROIs
    d1_ROIs={}
    d_coords[e]=d1_coords
    d1_coords={}

    
    
# Save ROIs and coordinates
dataset = open('Final_ROIs_dataset.pkl', 'wb')
pickle.dump(d_ROIs, dataset)
dataset.close()

dataset = open('Final_coords_dataset.pkl', 'wb')
pickle.dump(d_coords, dataset)
dataset.close()











