import sys

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

from encode_images import connect_to_mongodb, load_yaml


class SuperResolution:

    """
    Description: improves the quality of the infered images.
    Output: improved images.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")
        self.db, self.fs = connect_to_mongodb(config=self.config)

        self.DATASET = sys.argv[1]
        self.INPUT_FILTER = sys.argv[2]
        self.TARGET_FILTER = sys.argv[3]
        self.LEARNING_RATE = sys.argv[4]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.ENHANCEMENT_FACTOR = self.config["image_config"]["ENHANCEMENT_FACTOR"]

        self.improve("evolution")
        self.improve("inference")

    def improve(self, data_type):

        starting = f"{self.DATASET}_"
        ending = f"_{self.model_name}_{data_type}"

        imgs = [
            file.filename
            for file in self.fs.find(
                {"filename": {"$regex": f"^{starting}.*{ending}$"}}
            )
        ]

        for image in tqdm(imgs):

            file = self.fs.find_one({"filename": image})

            image_bytes = file.read()
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            data = image_array.reshape(self.IMAGE_DIM * 2, self.IMAGE_DIM * 2, 3)
            data = Image.fromarray(data)

            resized_element = data.resize(
                (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT), Image.ANTIALIAS
            )

            enhancer = ImageEnhance.Sharpness(resized_element)
            improved_element = enhancer.enhance(self.ENHANCEMENT_FACTOR)
            improved_array = np.array(improved_element)

            image_bytes = improved_array.tobytes()
            filename = f"{image}_super_resolution"
            self.fs.put(image_bytes, filename=filename)


if __name__ == "__main__":

    SuperResolution()
