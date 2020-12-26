import os
import glob
from pathlib import Path
from os import listdir
from numpy import asarray
from numpy import savez_compressed

import cv2
from keras.preprocessing.image import img_to_array
from moviepy.editor import *

from handlers import VideoClass, ImageClass

def video2frames(mode, paths, dim=(256, 256), trim_ratio=0.7):

    """
    
    Trims video and exports its frames.
    
    """

    Path(paths["trimmed"]).mkdir(parents=True, exist_ok=True)

    for file in glob.glob(f"{paths['raw']}/*"):
    
        video_name = os.path.splitext(file)[0].split('/')[-1]
        video_name = f"{video_name}.mp4"
        
        videohandler = VideoClass(input_path=file)
        videohandler.read_video()
        length = videohandler.get_video_length()
        end = length * trim_ratio
        start = length - end
        
        # Trim video
        video = VideoFileClip(file).cutout(start, end)
        trimmed_video_path = f"{paths['trimmed']}/{video_name}"
        video.write_videofile(trimmed_video_path)
        
        # Create frames
        videohandler = VideoClass(input_path=trimmed_video_path)
        videohandler.read_video()
        videohandler.get_video_name()
        videohandler.video2frames(output_path=paths["frames"], dim=dim)
