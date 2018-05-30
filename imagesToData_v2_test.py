import cv2
import os
import numpy as np

dir_path = './Taifeng_test1'
TEST_DATA_PATH = './testing_data'
none_dir_type = '.'
frame_type = '.jpg'
output_name = '.npy'

def main():
    train_list = []
    dirs = os.listdir(dir_path)
    dirs = [d for d in dirs if(none_dir_type not in d and 'gt' not in d)]
    for dir_name in dirs:
        print(dir_name)
        frames_source_path = dir_path + '/' + dir_name
        imgs = os.listdir(frames_source_path)
        imgs = [i for i in imgs if(frame_type in i)]
        #sort by frames
        imgs.sort(key = lambda x:int( x.split('.jpg')[0]))
        print(imgs)
        framelist = []

        for i in imgs:
            #print(i)
            image = cv2.imread(frames_source_path+'/'+i, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            framelist.append(np.float32(image/255.0))   
        framelist = np.array(framelist)
        np.save(TEST_DATA_PATH + '/' + 'test'+ dir_name + output_name, framelist) 
        #train_list.append(framelist)
    #train_list = np.array(train_list)
    #np.save(output_name, train_list)
if __name__ == '__main__':
    main()
