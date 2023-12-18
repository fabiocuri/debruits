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


class Preprocess:

    """
    Description: preprocessing of train, validation and test images.
    Output: .npz files with packed preprocessed images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.data = self.config["data"]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.image_path = f"./data/{self.data}/image/input"

        self.reset_folders()

        self.preprocess_data("train")
        self.preprocess_data("test")

    def reset_folders(self):

        for f in ["concat_images", "model_data"]:

            path = f"{self.image_path}/{f}"

            if os.path.exists(path):

                shutil.rmtree(path)

            os.mkdir(path)

    def extract_number(self, file_path):

        match = re.search(r"(\d+)\.png$", file_path)

        if match:

            return int(match.group(1))

        return 0

    def preprocess_data(self, mode):

        for file in tqdm(glob.glob(f"{self.image_path}/frames/{mode}/*")):

            input_img = ImageClass(config=self.config, input_path=file, mode=mode)
            input_img.read_image()
            input_img.resize((self.IMAGE_DIM, self.IMAGE_DIM))
            input_img.get_image_name()
            input_img.input_filter()

            target_img = ImageClass(config=self.config, input_path=file, mode=mode)
            target_img.read_image()
            target_img.resize((self.IMAGE_DIM, self.IMAGE_DIM))
            target_img.get_image_name()
            target_img.target_filter()

            imagehandler_concat = ImageClass(
                config=self.config,
                cv2image=cv2.hconcat([input_img.image, target_img.image]),
                mode=mode,
            )
            imagehandler_concat.read_image()
            imagehandler_concat.get_image_name(
                image_name=file.split("/")[-1].split(".")[0]
            )
            imagehandler_concat.export_image(
                output_path=f"{self.image_path}/concat_images/{mode}",
                scale=1,
            )

        sorted_file_paths = sorted(
            list(glob.glob(f"{self.image_path}/concat_images/{mode}/*")),
            key=self.extract_number,
        )

        src_list, tar_list = [], []

        for file in sorted_file_paths:

            imagehandler_frame = ImageClass(config=self.config, input_path=file)
            imagehandler_frame.read_image()
            pixels = img_to_array(imagehandler_frame.image)
            width = pixels.shape[1]
            src_list.append(pixels[:, : int(width / 2), :])
            tar_list.append(pixels[:, int(width / 2) :, :])

        [src_images_train, tar_images_train] = [asarray(src_list), asarray(tar_list)]

        savez_compressed(
            f"{self.image_path}/model_data/{mode}.npz",
            src_images_train,
            tar_images_train,
        )


if __name__ == "__main__":

    Preprocess()
