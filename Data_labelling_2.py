import pickle
import os
import cv2 as cv

# Open the final dataset of ROIs
pkl_file = open('Final_ROIs_dataset.pkl', 'rb')
final_dataset= pickle.load(pkl_file)
pkl_file.close()


# Create the folder which will contain the ROIs
final_data=input("Enter the path where you want to create the folder containing the final dataset: ")
try:
    os.chdir(final_data)
    os.mkdir("Final data")
    os.chdir(final_data+'/'+"Final data")
    os.mkdir("Frontal")
    os.mkdir("Lateral")
    os.chdir(final_data+'/'+"Final data"+'/'+"Frontal")
    os.mkdir("1")
    os.mkdir("2")
    os.mkdir("3")
    os.mkdir("4")
    os.mkdir("5")
    os.mkdir("6")
    os.chdir(final_data + '/' + "Final data" + '/' + "Lateral")
    os.mkdir("1")
    os.mkdir("2")
    os.mkdir("3")
    os.mkdir("4")
    os.mkdir("5")
    os.mkdir("6")
except:
    print("The folder already exists")


    
# Write the ROIs will OpenCV. This way, it will be easier to analyse them afterwards by OpenPose
for e in final_dataset:
    a=final_dataset[e]
    for j in a:
        b=a[j]
        for k in b:
            c=b[k]
            os.chdir(final_data + '/' + 'Final data' + '/' + e + '/' + k)
            if len(c)==3:
                cv.imwrite(j+'='+'.jpg',c[0])
                cv.imwrite(j +'=2'+ '.jpg', c[1])
                cv.imwrite(j + '=3'+'.jpg', c[0])
