import logging
import os
import sys
import cv2
from tqdm import tqdm

from mongodb_lib import (
    connect_to_mongodb,
    load_yaml
)

logging.basicConfig(level=logging.INFO)


def encode_images(fs):

    DATASET = sys.argv[1]

    for data_type in ["train", "test"]:

        images_dir = os.path.join(".", "data", data_type)
        files = list(os.listdir(images_dir))[:5]

        for filename in tqdm(files):

            if filename.lower().endswith((".jpg", ".jpeg")):

                image_path = os.path.join(images_dir, filename)
                image = cv2.imread(image_path)
                image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
                filename = f"{DATASET}_{data_type}_encoded_{filename}"
                fs.put(image_bytes, filename=filename)

        logging.info(
            f"Images in {data_type} encoded and saved to MongoDB successfully."
        )


def main():

    config = load_yaml(yaml_path="config_pipeline.yaml")
    _, fs = connect_to_mongodb(config=config)
    encode_images(fs=fs)


if __name__ == "__main__":

    main()
