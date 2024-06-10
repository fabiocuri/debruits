import sys

import cv2
from tqdm import tqdm

from mongodb_lib import load_yaml, connect_to_mongodb
import numpy as np
import os

class Frames2Videos:

    """
    Description: creates a video from image files.
    Output: a .mp4 file.
    """

    def __init__(self):

        self.MODE = sys.argv[1]
        self.DATASET = sys.argv[2]
        self.INPUT_FILTER = sys.argv[3]
        self.TARGET_FILTER = sys.argv[4]
        self.LEARNING_RATE = sys.argv[5]

        self.config = load_yaml("config_pipeline.yaml")

        self.FPS = self.config["video_config"]["FPS"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        if self.MODE == "jenkins":

            self.db, self.fs = connect_to_mongodb(config=self.config)

        self.create_video("evolution")
        self.create_video("inference")

    def extract_ids(self, image_name):

        parts = image_name.split("_")
        id1 = int(parts[3])
        id2 = int(parts[5])

        return id1, id2, image_name

    def create_video(self, data_type):

        if self.MODE == "jenkins":

            starting = f"{self.DATASET}_test_{data_type}_"
            ending = f"_{self.model_name}"

            imgs = [
                file.filename
                for file in self.fs.find(
                    {"filename": {"$regex": f"^{starting}.*{ending}$"}}
                )
            ]

        if self.MODE == "local":

            imgs = os.listdir(f"data/{data_type}")

        imgs = sorted(imgs, key=self.extract_ids)
        imgs = list(dict.fromkeys(imgs))

        temp_file_name = f"data/videos/{self.DATASET}_{data_type}_{self.model_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            temp_file_name,
            fourcc,
            self.FPS,
            (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT),
        )

        for img in tqdm(imgs):

            if self.MODE == "jenkins":

                file = self.fs.find_one({"filename": img})
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                data = nparr.reshape(self.IMAGE_DIM, self.IMAGE_DIM, 3).astype(np.uint8)

            if self.MODE == "local":

                data = cv2.imread(f"data/{data_type}/{img}")

            video_writer.write(data)

        video_writer.release()


if __name__ == "__main__":

    Frames2Videos()
