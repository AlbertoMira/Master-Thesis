# Imports
import shutil
import os


# Set the inputs
path_origin = input("Enter the path containing the ROIs: ") 
path_openpose = input("Enter the path of the OpenPose folder: ")
path_json = input("Enter the path where you want to store the keypoints coming from OpenPose: ")
path_json_folder_name = input("Enter the name of the folder you want to create in order to store those keypoints: ")
try: # Create the folder that will contain the keypoints
    os.mkdir(path_json+'/'+path_json_folder_name)
    os.chdir(path_json+'/'+path_json_folder_name)
    os.mkdir("Lateral")
    os.mkdir("Frontal")
    os.chdir(path_json+'/'+path_json_folder_name+'/'+'Frontal')
    os.mkdir("1")
    os.mkdir("2")
    os.mkdir("3")
    os.mkdir("4")
    os.mkdir("5")
    os.mkdir("6")
    os.chdir(path_json + '/' + path_json_folder_name + '/' + 'Lateral')
    os.mkdir("1")
    os.mkdir("2")
    os.mkdir("3")
    os.mkdir("4")
    os.mkdir("5")
    os.mkdir("6")

except:
    print('The folder already exists')

    
# Obtain the keypoints of the ROIs by using OpenPose and store them.
os.chdir(path_origin)
for e in os.listdir(path_origin): #Frontal, Lateral
    os.chdir(path_origin+'/'+e)
    for j in os.listdir(path_origin+'/'+e): # Pattern 1,2,3,4,5,6
        shutil.move(path_origin+'/'+e+'/'+j,path_openpose)
        current_path = path_origin+'/'+e+'/'+j
        os.chdir(path_openpose)
        os.system('bin\OpenPoseDemo.exe --image_dir '+j+' --render_pose 0 --display 0 --write_json keypoints output/')# --render_pose 0')
        shutil.move(path_openpose+'/'+j, path_origin+'/'+e)
        os.chdir(path_openpose+'/keypoints')
        for n in os.listdir(path_openpose+'/keypoints'):
            shutil.move(path_openpose+'/keypoints'+'/'+n, path_json+'/'+path_json_folder_name+'/'+e+'/'+j)
    os.chdir(path_origin)




