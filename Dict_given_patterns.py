import cv2 as cv
import os
import pickle
path = input("Enter the path of the folder containing the already-given patterns: ")
os.chdir(path)
d={}
d1={}
d2={}
d3={}
d4={}
d5={}
d6={}
l=[]
for e in os.listdir(path):
    os.chdir(path+'/'+e)
    for im in os.listdir(path+'/'+e):
        im1=im.split('_')
        name=im1[0]
        frame=im1[1]
        frame=frame[2]
        l.append(name)
        if frame =='1':
            try:
                img=cv.imread(path+'/'+e+'/'+im)
                img=cv.resize(img, (350,450))
                d1[name]=img
            except:
                print("Empty image")
        elif frame == '2':
            try:
                img = cv.imread(path + '/' + e + '/' + im)
                img = cv.resize(img, (350, 450))
                d2[name] = img
            except:
                print("Empty image")
        elif frame == '3':
            try:
                img = cv.imread(path + '/' + e + '/' + im)
                img = cv.resize(img, (350, 450))
                d3[name] = img
            except:
                print("Empty image")
        elif frame == '4':
            try:
                img = cv.imread(path + '/' + e + '/' + im)
                img = cv.resize(img, (350, 450))
                d4[name] = img
            except:
                print("Empty image")
        else:
            try:
                img = cv.imread(path + '/' + e + '/' + im)
                img = cv.resize(img, (350, 450))
                d5[name] = img
            except:
                print("Empty image")


    for j in l:
        try:
            l_dict=[]
            if len(d1[j])>0 and len(d2[j]>0) and len(d3[j]>0) and len(d4[j]>0) and len(d5[j]>0):
                l_dict.append(d1[j])
                l_dict.append(d2[j])
                l_dict.append(d3[j])
                l_dict.append(d4[j])
                l_dict.append(d5[j])
                d6[j]=l_dict
                l_dict=[]

        except:
            print("Dictionaries not filled")


    d[e]=d6
    d1={}
    d2={}
    d3={}
    d4={}
    d5={}
    d6={}



#Given patterns contains the information related to the already-given patterns
given_patterns = open('Skating_1_1_given_patterns.pkl', 'wb')
pickle.dump(d, given_patterns)
given_patterns.close()