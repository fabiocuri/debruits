import base64
import logging
from io import BytesIO

import numpy as np
import yaml
from gridfs import GridFS
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)


def load_yaml(yaml_path):

    with open(yaml_path, "r") as file:

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

        fs = GridFS(db)

        return db, fs

    except Exception as e:

        logging.error(f"Failed to connect to MongoDB: {e}")

        raise


def delete_collection(db, collection_name):

    collection = db[collection_name]
    collection.delete_many({})


def save_image_to_mongodb(image_data, filename, collection):

    base64_image = base64.b64encode(image_data)
    base64_image_str = base64_image.decode("utf-8")
    image_doc = {
        "filename": filename,
        "base64_image": base64_image_str,
    }
    collection.insert_one(image_doc)


def load_data_from_chunks(fs, id_name, db):

    file = fs.find_one({"filename": id_name})
    chunks_cursor = db.fs.chunks.find({"files_id": file._id}).sort("n", 1)
    data_chunks = b"".join(chunk["data"] for chunk in chunks_cursor)
    data = np.load(BytesIO(data_chunks))

    return data


def preprocess_chunks(fs, id_name, db):

    data = load_data_from_chunks(fs, id_name, db)

    X1, X2 = data["arr_0"], data["arr_1"]

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]


def save_model(fs, model_object, model_object_name):

    model_bytes = model_object.to_json().encode()
    fs.put(model_bytes, filename=model_object_name)
