import io
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from image import ImageClass
from mongodb_lib import connect_to_mongodb, load_image_from_chunks, load_yaml


def preprocess_data():

    MODE = sys.argv[1]
    DATASET = sys.argv[2]
    INPUT_FILTER = sys.argv[3]
    TARGET_FILTER = sys.argv[4]
    LEARNING_RATE = sys.argv[5]

    model_name = f"{INPUT_FILTER}_{TARGET_FILTER}_{LEARNING_RATE}"

    if MODE == "jenkins":

        config = load_yaml(yaml_path="config_pipeline.yaml")
        _, fs = connect_to_mongodb(config=config)

        for data_type in ["train", "test"]:

            filename = f"{DATASET}_{data_type}_preprocessed_{model_name}"

            src_list, tar_list = [], []

            starting = f"{DATASET}_{data_type}_encoded_"

            imgs = [
                file.filename
                for file in fs.find({"filename": {"$regex": f"^{starting}.*"}})
            ]

            for index, img in enumerate(tqdm(imgs)):

                image = load_image_from_chunks(fs, img)

                input_img = ImageClass(image=image)
                input_img.input_filter(INPUT_FILTER)

                target_img = ImageClass(image=image)
                target_img.target_filter(TARGET_FILTER)

                src_list.append(input_img.image)
                tar_list.append(target_img.image)

                if index == 0:

                    cv2.imwrite(
                        "source_example.png", (input_img.image).astype(np.uint8)
                    )
                    cv2.imwrite(
                        "target_example.png", (target_img.image).astype(np.uint8)
                    )

            src_images_train = np.stack(src_list)
            tar_images_train = np.stack(tar_list)

            npz_data = io.BytesIO()
            np.savez_compressed(npz_data, src_images_train, tar_images_train)

            npz_data.seek(0)

            fs.put(npz_data, filename=filename)

    if MODE == "local":

        for data_type in ["train", "test"]:

            filename = f"{DATASET}_{data_type}_preprocessed_{model_name}"

            src_list, tar_list = [], []

            data_path = f"data/{data_type}"

            imgs = os.listdir(data_path)

            for index, img in enumerate(tqdm(imgs)):

                image = cv2.imread(f"{data_path}/{img}")

                input_img = ImageClass(image=image)
                input_img.input_filter(INPUT_FILTER)

                target_img = ImageClass(image=image)
                target_img.target_filter(TARGET_FILTER)

                src_list.append(input_img.image)
                tar_list.append(target_img.image)

                if index == 0:

                    cv2.imwrite(
                        "source_example.png", (input_img.image).astype(np.uint8)
                    )
                    cv2.imwrite(
                        "target_example.png", (target_img.image).astype(np.uint8)
                    )

            src_images_train = np.stack(src_list)
            tar_images_train = np.stack(tar_list)

            np.savez_compressed(f"data/{filename}", src_images_train, tar_images_train)


def main():

    preprocess_data()


if __name__ == "__main__":

    main()
