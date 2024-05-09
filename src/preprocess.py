import io
import sys
import numpy as np
from tqdm import tqdm
import cv2
from image import ImageClass
from mongodb_lib import load_yaml, connect_to_mongodb, load_image_from_chunks


def preprocess_data(fs):

    DATASET = sys.argv[1]
    INPUT_FILTER = sys.argv[2]
    TARGET_FILTER = sys.argv[3]
    LEARNING_RATE = sys.argv[4]

    model_name = f"{INPUT_FILTER}_{TARGET_FILTER}_{LEARNING_RATE}"

    for data_type in ["train", "test"]:

        filename = f"{DATASET}_{data_type}_preprocessed_{model_name}"

        if fs.find_one({"filename": filename}) is None:

            src_list, tar_list = [], []

            starting = f"{DATASET}_{data_type}_encoded_"

            imgs = [
                file.filename
                for file in fs.find(
                    {"filename": {"$regex": f"^{starting}.*"}}
                )
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

                    cv2.imwrite("source_example.png", (input_img.image).astype(np.uint8))
                    cv2.imwrite("target_example.png", (target_img.image).astype(np.uint8))

            src_images_train = np.stack(src_list)
            tar_images_train = np.stack(tar_list)

            npz_data = io.BytesIO()
            np.savez_compressed(npz_data, src_images_train, tar_images_train)

            npz_data.seek(0)

            fs.put(npz_data, filename=filename)


def main():

    config = load_yaml(yaml_path="config_pipeline.yaml")
    _, fs = connect_to_mongodb(config=config)
    preprocess_data(fs=fs)


if __name__ == "__main__":

    main()
