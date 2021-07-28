import os
import json
import filecmp
import numpy as np

#os.chdir(r'C:/Users/Albert/Desktop/openpose/')
#os.system('cd Desktop')
#a='keypoints'
#os.system('bin\OpenPoseDemo.exe --image_dir frames --write_json '+a+' output/ --display 0 --render_pose 0')
#os.system('bin\OpenPoseDemo.exe --image_dir Nueva carpeta --write_json '+a+' output/')

# Open the JSON files
# with open(r'C:\Users\Albert\Desktop\keypoints\Frontal\Pattern_1\BAIER Konstantin Pattern 1_keypoints.json', 'r') as myfile:
    #data=myfile.read()
# obj = json.loads(data)
# obj=obj['people']
#d={}
#d=obj[0]
#print(d)
#l=d['pose_keypoints_2d']
#print(l)
import pickle

d={}
d1={}
d2={}

pkl_file_2 = open('Final_coords_dataset.pkl', 'rb')
coord= pickle.load(pkl_file_2)
pkl_file_2.close()

keypoints = input("Enter the path of the folder containing the keypoints from OpenPose: ")

os.chdir(keypoints)
for v in os.listdir(keypoints): #Lateral, Frontal
    os.chdir(keypoints+'/'+v)
    for w in os.listdir(keypoints+'/'+v): # 1,2,3,4,5,6
        os.chdir(keypoints+'/'+v+'/'+w)
        l_matrix=[]
        #current_pattern = w[len(w) - 1] # 1,2,3,4,5,6
        for n in os.listdir(keypoints+'/'+v+'/'+w):
            with open(keypoints+'/'+v+'/'+w+'/'+n,'r') as myfile:
                data = myfile.read()
            obj = json.loads(data)
            obj = obj['people']
            #a=0

            current_name = n.split(".")
            current_name=current_name[0] #Baier Konstantin.2_keypoints
            current_name=''.join(current_name)
            current_name=current_name.split() #[Baier, Konstantin=2_keypoints]
            #Get surname
            current_name_1=current_name[0]

            #Get name
            current_name_2=''.join(current_name[1])
            current_name_2=current_name_2.split("=")
            current_name_3=current_name_2[0]


            #Identify which one of the 3 images of the athlete w.r.t the frame we are working on
            print(current_name)
            current_name_2=current_name_2[1]
            current_name_2=''.join(current_name_2)
            current_name_2=current_name_2[0]

            current_name=current_name_1+" "+current_name_3 # Surname + name of the athlete

            coord_2 = coord[v] # Get either the lateral or frontal coordinates
            coord_2 = coord_2[current_name] # Get the either the lateral or frontal coordinates of an athlete
            coord_2 = coord_2[w] # Get the list of coordinates of a single pattern

            if len(obj) > 1:
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
                    # print(no_0,no_1,no_2,no_3, no_4, no_5, no_6, no_7, no_8, no_9, no_10, no_11, no_12, no_13, no_14, no_15, no_16, no_17, no_18, no_19, no_20, no_21, no_22, no_23, no_24)
                    # pers = 0
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

                if current_name_2=='_':
                    coord_2=coord_2[0]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
                elif current_name_2=='2':
                    coord_2 = coord_2[1]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
                elif current_name_2=='3':
                    coord_2 = coord_2[2]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)

            elif len(obj)==1:
                d = obj[0]
                l = d['pose_keypoints_2d']
                i = 0
                arr = np.zeros((1, 2))
                while i < len(l):
                    if len(arr) == 1 and arr[0][0]==0 and arr[0][1]==0:
                        arr[0][0] = l[i]
                        arr[0][1] = l[i + 1]
                    else:
                        arr = np.append(arr, [[l[i], l[i + 1]]], axis=0)
                    i += 3
                if current_name_2=='_':
                    coord_2=coord_2[0]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
                elif current_name_2=='2':
                    coord_2 = coord_2[1]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
                elif current_name_2=='3':
                    coord_2 = coord_2[2]
                    arr = np.append(arr, [[coord_2[0], coord_2[1]]], axis=0)
                    arr = np.append(arr, [[coord_2[2], coord_2[3]]], axis=0)
            if len(obj)>=1:
                l_matrix.append(arr)
        d1[w]=l_matrix
    d2[v]=d1
    d1={}


os.chdir(r'C:\Users\alberto.mira\PycharmProjects\pythonProject')
colab = open('COLAB_ahora.pkl', 'wb')
pickle.dump(d2, colab)
colab.close()


