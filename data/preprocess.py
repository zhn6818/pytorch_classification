import os
import glob
import numpy



import sys 
# print("~~~~~~~~~~~",sys.path, "~~~~~~~~~~~")
sys.path.append("/data1/zhn/2022/pytorch_classification") 
# print("!!!!!!!!!!!!!!!!!",sys.path, "~~~~~~~~~~~")
import cfg
import random



if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = cfg.BASE + 'val'
    ##写train.txt文件
    txtpath = cfg.BASE
    # print(labels)
    imglist = []
    for index, label in enumerate(labels):
        imgli = glob.glob(os.path.join(traindata_path,label, '*.jpg'))
        imglist.extend(imgli)
        # print(imglist)
    numpy.random.shuffle(imglist)
    print(len(imglist))
    trainlist = imglist[:int(0.8*len(imglist))]
    vallist = imglist[(int(0.8*len(imglist))+1):]
    with open(txtpath + 'train.txt', 'a')as f:
        for img in trainlist:
            # print(img + ' ' + str(index))
            f.write(img + ' ' + str(labels.index(img.split('/')[-2])))
            f.write('\n')

    with open(txtpath + 'val.txt', 'a')as f:
        for img in vallist:
            # print(img + ' ' + str(index))
            f.write(img + ' ' + str(labels.index(img.split('/')[-2])))
            f.write('\n')


    imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    with open(txtpath + 'test.txt', 'a')as f:
        for img in imglist:
            f.write(img)
            f.write('\n')