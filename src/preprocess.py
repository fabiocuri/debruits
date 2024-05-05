import base64
import io

import numpy as np
from PIL import Image
from tqdm import tqdm

from image import ImageClass
from mongodb_lib import connect_to_mongodb, load_yaml


def preprocess_data(db, fs, yaml_data):

    config = load_yaml(yaml_path="config.yaml")

    INPUT_FILTER = config["model_config"]["INPUT_FILTER"]
    TARGET_FILTER = config["model_config"]["TARGET_FILTER"]
    LEARNING_RATE = config["model_config"]["LEARNING_RATE"]

    model_name = f"{INPUT_FILTER}_{TARGET_FILTER}_{LEARNING_RATE}"

    for data_type in ["train", "test"]:

        filename = f"{data_type}_preprocessed_{model_name}"

        if fs.find_one({"filename": filename}) is None:

            src_list, tar_list = [], []

            collection = db[yaml_data[f"mongoDb{data_type}Collection"]]

            cursor = list(collection.find({}))

            for document in tqdm(cursor):

                base64_image_str = document.get("base64_image", "")
                image_bytes = base64.b64decode(base64_image_str)
                image_stream = Image.open(io.BytesIO(image_bytes))
                image = np.array(image_stream)

                input_img = ImageClass(image=image)
                input_img.input_filter()

                target_img = ImageClass(image=image)
                target_img.target_filter()

                src_list.append(input_img.image)
                tar_list.append(target_img.image)

            src_images_train = np.stack(src_list)
            tar_images_train = np.stack(tar_list)

            npz_data = io.BytesIO()
            np.savez_compressed(npz_data, src_images_train, tar_images_train)

            npz_data.seek(0)

            fs.put(npz_data, filename=filename)


def main():

    yaml_data = load_yaml(yaml_path="debruits-kubernetes/values.yaml")
    db, fs = connect_to_mongodb(yaml_data=yaml_data)
    preprocess_data(db=db, fs=fs, yaml_data=yaml_data)


if __name__ == "__main__":

    main()
