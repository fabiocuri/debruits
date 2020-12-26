import os
import glob
from pathlib import Path
from os import listdir
from numpy import asarray
from numpy import savez_compressed

import cv2
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from moviepy.editor import *

from handlers import ImageClass


def frames2canny(mode, paths):

    """
    
    Creates canny edges from the frames.
    
    """

    for file in glob.glob(f"{paths['frames']}/{mode}/*"):

        imagehandler = ImageClass(input_path=file, mode=mode)
        imagehandler.read_image()
        imagehandler.resize((256, 256))
        imagehandler.get_image_name()
        imagehandler.export_image(output_path=f"{paths['resized']}/{mode}")
        imagehandler.edges_canny()
        imagehandler.export_image(output_path=f"{paths['edges']}/{mode}")


def concat2model(mode, paths):

    """
    
    Concatenates source and target images.
    
    """

    for file in glob.glob(f"{paths['edges']}/{mode}/*"):
    
        image_name = file.split('/')[-1]
    
        imagehandler_frame = ImageClass(input_path=f"{paths['resized']}/{mode}/{image_name}", mode=mode)
        imagehandler_frame.read_image()

        imagehandler_edges = ImageClass(input_path=file, mode=mode)
        imagehandler_edges.read_image()

        concat_img = cv2.hconcat(
            [imagehandler_edges.image, imagehandler_frame.image])

        imagehandler_concat = ImageClass(cv2image=concat_img, mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=image_name)
        imagehandler_concat.export_image(output_path=f"{paths['model']}/{mode}")


def load_images(mode, paths):

    """
    
    Creates train and val zipped data.
    
    """

    src_list, tar_list = list(), list()

    for file in glob.glob(f"{paths['model']}/{mode}/*"):

        imagehandler_frame = ImageClass(input_path=file)
        imagehandler_frame.read_image()
        pixels = img_to_array(imagehandler_frame.image)
        
        width = pixels.shape[1]

        edges_img, orig_img = pixels[:, :int(width/2), :], pixels[:, int(width/2):, :]
        src_list.append(edges_img)
        tar_list.append(orig_img)

    return [asarray(src_list), asarray(tar_list)]


def preprocess4GAN(mode, paths):

    frames2canny(mode=mode, paths=paths)
    
    concat2model(mode=mode, paths=paths)

    [src_images_train, tar_images_train] = load_images(mode=mode, paths=paths)
    savez_compressed(f"{paths['model']}/{mode}.npz",
                     src_images_train, tar_images_train)


if __name__ == "__main__":

    input_path = "../../../data/input"
    
    paths = {"frames": f"{input_path}/frames",
            "resized": f"{input_path}/resized",
            "edges": f"{input_path}/edges",
            "model": f"{input_path}/model",
            "test": f"{input_path}/test"}

    preprocess4GAN(mode='train', paths=paths)
    preprocess4GAN(mode='val', paths=paths)
    preprocess4GAN(mode='test', paths=paths)
