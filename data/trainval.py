import os
import shutil
 
file_dir = "/data1/zhn/data1/train/train"
path = "/data1/zhn/data1"
 
#将某类图片移动到该类的文件夹下
def img_to_file(path):
    print("=========开始移动图片============")
     #如果没有dog类和cat类文件夹，则新建
    if not os.path.exists(path+"/dog"):
            os.makedirs(path+"/dog")
    if not os.path.exists(path+"/cat"):
            os.makedirs(path+"/cat")
    file_name_list = os.listdir(file_dir)
    print("共：{}张图片".format(len(file_name_list)))
    for imgName in file_name_list:
        # 去除后缀
        img = imgName.replace(".jpg","")
         #将图片移动到指定的文件夹中
        if img.split(".")[0] == "cat":
            shutil.move(file_dir+"/"+imgName,path+"/cat")
        if img.split(".")[0] == "dog":
            shutil.move(file_dir+"/"+imgName,path+"/dog")
    print("=========移动图片完成============")    
img_to_file(path)