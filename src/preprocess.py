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


def frames2canny(mode, paths):

    """
    
    Creates canny edges from the frames.
    
    """

    for file in glob.glob(f"{paths['frames']}/*/*"):

        imagehandler = ImageClass(input_path=file)
        imagehandler.read_image()
        imagehandler.get_image_name()
        imagehandler.edges_canny()

        # Exports image if not too dark
        if cv2.countNonZero(imagehandler.image) != 0:

            imagehandler.export_image(
                output_path=paths['edges'])


def concat2model(mode, paths):

    """
    
    Concatenates source and target images.
    
    """

    frames_files = ['/'.join(file.split('/')[-2:]) for file in glob.glob(f"{paths['edges']}/*/*")]

    for frames_file in frames_files:

        imagehandler_frame = ImageClass(
            input_path=f"{paths['frames']}/{frames_file}")
        imagehandler_frame.read_image()

        imagehandler_edges = ImageClass(
            input_path=f"{paths['edges']}/{frames_file}")
        imagehandler_edges.read_image()

        concat_img = cv2.hconcat(
            [imagehandler_edges.image, imagehandler_frame.image])
        imagehandler_concat = ImageClass(cv2image=concat_img)
        imagehandler_concat.read_image()
        image_name = frames_file.replace('/', '_')
        imagehandler_concat.get_image_name(image_name=image_name)
        imagehandler_concat.export_image(output_path=paths['model'])


def load_images(path):

    src_list, tar_list = list(), list()

    for filename in listdir(path):

        imagehandler_frame = ImageClass(input_path=f"{path}/{filename}")
        imagehandler_frame.read_image()
        pixels = img_to_array(imagehandler_frame.image)

        edges_img, orig_img = pixels[:, :256, :], pixels[:, 256:, :]
        src_list.append(edges_img)
        tar_list.append(orig_img)

    return [asarray(src_list), asarray(tar_list)]


def preprocess4GAN(mode, paths):

    video2frames(mode=mode, paths=paths)
    frames2canny(mode=mode, paths=paths)
    concat2model(mode=mode, paths=paths)

    [src_images_train, tar_images_train] = load_images(paths["model"])
    savez_compressed(f"{paths['model']}/{mode}_256.npz",
                     src_images_train, tar_images_train)


if __name__ == "__main__":

    input_path = "../../../data/input"

    for mode in ['train', 'val']:
    
        paths = {"raw": f"{input_path}/raw/{mode}",
                "trimmed": f"{input_path}/trimmed/{mode}",
                "frames": f"{input_path}/frames/{mode}",
                "edges": f"{input_path}/edges/{mode}",
                "model": f"{input_path}/model/{mode}"}

        preprocess4GAN(mode=mode, paths=paths)
        #preprocess4GAN(mode=mode, paths=paths)
