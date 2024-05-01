import base64
import logging
import os

import yaml
from pymongo import MongoClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def connect_to_mongodb(yaml_data):
    try:
        mongodb_username = base64.b64decode(yaml_data["mongodbUsernameBase64"]).decode(
            "utf-8"
        )
        mongodb_password = base64.b64decode(yaml_data["mongodbPasswordBase64"]).decode(
            "utf-8"
        )
        mongodb_port = str(yaml_data["mongoDbPort"])
        client = MongoClient(
            f"mongodb://{mongodb_username}:{mongodb_password}@localhost:{mongodb_port}/?authSource=admin"
        )
        db = client[yaml_data["mongoDbDatabase"]]
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise


def delete_all_documents_in_collections(db):
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        collection.delete_many({})


def encode_images(db, yaml_data):
    for data_type in ["train", "test"]:
        collection = db[yaml_data[f"mongoDb{data_type}Collection"]]
        images_dir = os.path.join(".", "data", data_type)
        for filename in tqdm(os.listdir(images_dir)):
            if filename.lower().endswith((".jpg", ".jpeg")):
                image_path = os.path.join(images_dir, filename)
                try:
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data)
                        base64_image_str = base64_image.decode("utf-8")
                        image_doc = {
                            "filename": filename,
                            "base64_image": base64_image_str,
                        }
                        collection.insert_one(image_doc)
                except Exception as e:
                    logging.error(f"Failed to encode and save image {filename}: {e}")
                    continue
        logging.info(
            f"Images in {data_type} encoded and saved to MongoDB successfully."
        )


def main():
    try:
        yaml_data = load_yaml("./debruits-kubernetes/values.yaml")
        db = connect_to_mongodb(yaml_data)
        delete_all_documents_in_collections(db)
        encode_images(db, yaml_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
