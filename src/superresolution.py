import os
import sys

import cv2
import yaml
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import numpy as np

class SuperResolution:

    """
    Description: improves quality of the images.
    Output: improved images.
    """

    def __init__(self):

        self.config = yaml.load(open("config_pipeline.yaml"), Loader=yaml.FullLoader)

        self.IMAGES_FOLDER = sys.argv[1]

        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.output_path = "data/super_resolution"
        os.makedirs(self.output_path, exist_ok=True)

        self.improve_quality()

    def improve_quality(self):

        imgs = os.listdir(self.IMAGES_FOLDER)

        for img in tqdm(imgs, total=len(imgs)):

            data = Image.open(f"{self.IMAGES_FOLDER}/{img}")
            data = np.array(data)

            data = cv2.GaussianBlur(data, (5, 5), 0)

            data = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(data)
            cv2.equalizeHist(channels[0], channels[0])
            data = cv2.merge(channels)
            data = cv2.cvtColor(data, cv2.COLOR_YCrCb2BGR)

            data = cv2.GaussianBlur(data, (1, 1), 0)

            data = cv2.resize(
                data,
                (self.IMAGE_DIM, self.IMAGE_DIM),
                interpolation=cv2.INTER_LINEAR,
            )

            cv2.imwrite(f"{self.output_path}/{img}", data)


if __name__ == "__main__":

    SuperResolution()
