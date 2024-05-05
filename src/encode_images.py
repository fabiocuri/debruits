import logging
import os

from tqdm import tqdm

from mongodb_lib import (
    connect_to_mongodb,
    delete_collection,
    load_yaml,
    save_image_to_mongodb,
)

logging.basicConfig(level=logging.INFO)


def encode_images(db, yaml_data):

    for data_type in ["train", "test"]:

        collection_name = yaml_data[f"mongoDb{data_type}Collection"]
        delete_collection(db=db, collection_name=collection_name)

        collection = db[collection_name]
        images_dir = os.path.join(".", "data", data_type)
        files = list(os.listdir(images_dir))[:5]

        for filename in tqdm(files):

            if filename.lower().endswith((".jpg", ".jpeg")):

                image_path = os.path.join(images_dir, filename)

                with open(image_path, "rb") as image_file:

                    image_data = image_file.read()
                    save_image_to_mongodb(
                        image_data=image_data, filename=filename, collection=collection
                    )

        logging.info(
            f"Images in {data_type} encoded and saved to MongoDB successfully."
        )


def main():

    yaml_data = load_yaml(yaml_path="debruits-kubernetes/values.yaml")
    db, _ = connect_to_mongodb(yaml_data=yaml_data)
    encode_images(db=db, yaml_data=yaml_data)


if __name__ == "__main__":

    main()
