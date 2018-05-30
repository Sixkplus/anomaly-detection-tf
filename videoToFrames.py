import cv2
import os
import numpy as np

video_source_path = './training_videos'
video_type = '.avi'

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'i.png' where 
    # i is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    if(not os.path.isdir(path_output_dir)):
        os.mkdir(path_output_dir)
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % count, image)
            count += 1
        else:
            break
    #cv2.destroyAllWindows()
    vidcap.release()

def main():
    videos=os.listdir(video_source_path)
    videos = [v for v in videos if(video_type in v)]
    videos.sort(key = lambda x:int(x.split(video_type)[0]))
    for v in videos:
        print(v)
        video_to_frames(video_source_path + '/' + v, video_source_path + '/' + v.replace('.avi',''))


if __name__ == '__main__':
    main()

