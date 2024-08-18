import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from src_super_resolution.rdn import RDN
from src_super_resolution.rrdn import RRDN


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
        # model = RRDN(weights='gans')
        # model_gan = RRDN(weights="gans")
        model_noise = RDN(weights="noise-cancel")
        rdn = RDN(arch_params={"C": 3, "D": 10, "G": 64, "G0": 64, "x": 2})
        rdn.model.load_weights("weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5")

        for img in tqdm(imgs, total=len(imgs)):

            data = Image.open(f"{self.IMAGES_FOLDER}/{img}")
            data = np.array(data)
            data = cv2.resize(
                data, (self.IMAGE_DIM, self.IMAGE_DIM), interpolation=cv2.INTER_LINEAR
            )

            for i in range(2):

                data = rdn.predict(data)
                data = cv2.resize(
                    data,
                    (self.IMAGE_DIM, self.IMAGE_DIM),
                    interpolation=cv2.INTER_LINEAR,
                )

            data = model_noise.predict(data)
            data = cv2.resize(
                data, (self.IMAGE_DIM, self.IMAGE_DIM), interpolation=cv2.INTER_LINEAR
            )

            for i in range(2):

                data = rdn.predict(data)
                data = cv2.resize(
                    data,
                    (self.IMAGE_DIM, self.IMAGE_DIM),
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imwrite(f"{self.output_path}/{img}", data)


if __name__ == "__main__":

    SuperResolution()
