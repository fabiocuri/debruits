import cv2
import os

class VideoClass:
    """ 
    A class that reads, preprocesses and converts videos.
    """

    def __init__(self, video_path='./', frames_path='./'):

        super(VideoClass, self).__init__()
        self.video_path = video_path
        self.frames_path = frames_path

    def read_video(self):
        self.video = cv2.VideoCapture(self.video_path)

    def get_video_name(self):
        base = os.path.basename(self.video_path)
        self.video_name = os.path.splitext(base)[0]

    def video2frames(self):

        self.read_video()
        self.get_video_name()

        success, image = self.video.read()
        count = 1
        
        print("{}/frame_{}_{}.jpg".format(self.frames_path, count, self.video_name))

        while success:

            cv2.imwrite("{}/frame_{}_{}.jpg".format(self.frames_path, count, self.video_name), image)    
            success, image = self.video.read()
            print('Saved image ', count)
            count += 1