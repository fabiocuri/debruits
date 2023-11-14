import glob
import os
import re
import shutil

import cv2
import yaml
from numpy import asarray, savez_compressed
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

from handlers import ImageClass


def create_model_data(config, mode, paths):
    for file in tqdm(glob.glob(f"{paths['frames']}/{mode}/*")):

        input_img = ImageClass(config=config, input_path=file, mode=mode)
        input_img.read_image()
        input_img.resize((1024, 1024))
        input_img.get_image_name()
        input_img.input_filter()
        input_img.export_image(output_path=f"{paths['input_images']}/{mode}", scale=255)

        target_img = ImageClass(config=config, input_path=file, mode=mode)
        target_img.read_image()
        target_img.resize((1024, 1024))
        target_img.get_image_name()
        target_img.target_filter()
        target_img.export_image(
            output_path=f"{paths['target_images']}/{mode}", scale=255
        )

        imagehandler_concat = ImageClass(
            config=config,
            cv2image=cv2.hconcat([input_img.image, target_img.image]),
            mode=mode,
        )
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=file.split("/")[-1].split(".")[0])
        imagehandler_concat.export_image(
            output_path=f"{paths['concat_images']}/{mode}", scale=1
        )


def load_images(config, mode, paths):
    def extract_number(file_path):
        match = re.search(r"(\d+)\.png$", file_path)
        if match:
            return int(match.group(1))
        return 0

    sorted_file_paths = sorted(
        list(glob.glob(f"{paths['concat_images']}/{mode}/*")), key=extract_number
    )

    src_list, tar_list = [], []
    for file in tqdm(sorted_file_paths):
        imagehandler_frame = ImageClass(config=config, input_path=file)
        imagehandler_frame.read_image()
        pixels = img_to_array(imagehandler_frame.image)
        width = pixels.shape[1]
        src_list.append(pixels[:, : int(width / 2), :])
        tar_list.append(pixels[:, int(width / 2) :, :])
    return [asarray(src_list), asarray(tar_list)]


def preprocess4GAN(mode, config):
    paths = {
        "frames": f"./data/{config['folder']}/image/input/frames",
        "input_images": f"./data/{config['folder']}/image/input/input_images",
        "target_images": f"./data/{config['folder']}/image/input/target_images",
        "concat_images": f"./data/{config['folder']}/image/input/concat_images",
        "model_data": f"./data/{config['folder']}/image/input/model_data",
    }
    create_model_data(config=config, mode=mode, paths=paths)
    [src_images_train, tar_images_train] = load_images(
        config=config, mode=mode, paths=paths
    )
    savez_compressed(
        f"{paths['model_data']}/{mode}.npz", src_images_train, tar_images_train
    )


if __name__ == "__main__":

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)
    folder = config["folder"]

    for f in ["input_images", "target_images", "concat_images", "model_data"]:

        path = f"./data/{folder}/image/input/{f}"

        if os.path.exists(path):

            shutil.rmtree(path)

        os.mkdir(path)


    for mode in ["train", "val", "test"]:
        preprocess4GAN(mode=mode, config=config)
