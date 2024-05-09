import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from encode_images import connect_to_mongodb, load_yaml


class Frames2Videos:

    """
    Description: creates a video from image files.
    Output: a .mp4 file.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")
        self.db, self.fs = connect_to_mongodb(config=self.config)

        self.DATASET = sys.argv[1]
        self.INPUT_FILTER = sys.argv[2]
        self.TARGET_FILTER = sys.argv[3]
        self.LEARNING_RATE = sys.argv[4]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        self.FPS = self.config["video_config"]["FPS"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.create_video("evolution")
        self.create_video("inference")

    def extract_ids(self, image_name):

        parts = image_name.split("_")
        id1 = int(parts[3])
        id2 = int(parts[5])

        return id1, id2, image_name

    def create_video(self, data_type):

        starting = f"{self.DATASET}_test_{data_type}_"
        ending = ending = f"_{self.model_name}"

        imgs = [
            file.filename
            for file in self.fs.find(
                {"filename": {"$regex": f"^{starting}.*{ending}$"}}
            )
        ]

        imgs = sorted(imgs, key=self.extract_ids)
        imgs = list(dict.fromkeys(imgs))

        temp_file_name = f"{starting}{ending}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            temp_file_name,
            fourcc,
            self.FPS,
            (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT),
        )

        for img in tqdm(imgs):

            grid_out = self.fs.find_one({"filename": img})
            image_bytes = grid_out.read()
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            print(image_array.shape)
            data = image_array.reshape(self.ENHANCED_HEIGHT, self.ENHANCED_WIDTH, 3)
            #data = Image.fromarray(data)
            #data = np.array(data.convert("RGB"))[:, :, ::-1]
            video_writer.write(data)

        video_writer.release()


if __name__ == "__main__":

    Frames2Videos()
