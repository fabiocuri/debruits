import os
import random
from pathlib import Path

import cv2
import yaml


class Overlap:

    """
    Description: overlaps inference images of different runs.
    Output: overlapped images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.data = self.config["data"]
        self.overlap_folders = self.config["overlap-images"]
        self.folder_1 = self.overlap_folders[0]
        self.folder_2 = self.overlap_folders[1]

        self.images_path = f"./data/{self.data}/image/output/"
        self.output_path = (
            f"{self.images_path}/overlap_{self.folder_1}_{self.folder_2}/"
        )

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.overlap()

    def overlap(self):

        for file in list(range(100)):

            file = random.choice(
                os.listdir(f"{self.images_path}/{self.folder_1}/inference/")
            )

            image_0 = cv2.imread(f"{self.images_path}/{self.folder_1}/inference/{file}")

            file = random.choice(
                os.listdir(f"{self.images_path}/{self.folder_2}/inference/")
            )

            image_1 = cv2.imread(f"{self.images_path}/{self.folder_1}/inference/{file}")

            overlap = cv2.addWeighted(image_0, 1, image_1, 1, 0)

            cv2.imwrite(f"{self.output_path}/{file}.png", overlap)


if __name__ == "__main__":

    Overlap()
