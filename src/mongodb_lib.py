import base64
import logging
from io import BytesIO

import numpy as np
import yaml
from gridfs import GridFS
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
import cv2


def load_yaml(yaml_path):

    with open(yaml_path, "r") as file:

        return yaml.safe_load(file)


def connect_to_mongodb(config):

    try:

        mongodb_username = base64.b64decode(config["mongodbUsernameBase64"]).decode(
            "utf-8"
        )
        mongodb_password = base64.b64decode(config["mongodbPasswordBase64"]).decode(
            "utf-8"
        )
        cluster_node_id = str(config["clusterNodeId"])
        mongodb_nodeport_port = str(config["mongoDbNodeportPort"])
        client = MongoClient(
            f"mongodb://{mongodb_username}:{mongodb_password}@{cluster_node_id}:{mongodb_nodeport_port}/?authSource=admin"
        )
        db = client[config["mongoDbDatabase"]]

        fs = GridFS(db)

        return db, fs

    except Exception as e:

        logging.error(f"Failed to connect to MongoDB: {e}")

        raise


def load_image_from_chunks(fs, filename):

    file = fs.find_one({"filename": filename})
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


def preprocess_npz(fs, db, filename):

    file = fs.find_one({"filename": filename})
    chunks_cursor = db.fs.chunks.find({"files_id": file._id}).sort("n", 1)
    data_chunks = b"".join(chunk["data"] for chunk in chunks_cursor)
    data = np.load(BytesIO(data_chunks))

    X1, X2 = data["arr_0"], data["arr_1"]

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    X1 = np.clip(X1 * 255, 0, 255).astype(np.uint8)
    X2 = np.clip(X2 * 255, 0, 255).astype(np.uint8)

    return [X1, X2]


def preprocess_npz_local(filename):

    data = np.load(filename)
    X1, X2 = data["arr_0"], data["arr_1"]

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    X1 = np.clip(X1 * 255, 0, 255).astype(np.uint8)
    X2 = np.clip(X2 * 255, 0, 255).astype(np.uint8)

    return [X1, X2]


def save_model(fs, model_object, model_object_name):

    model_bytes = model_object.to_json().encode()
    fs.put(model_bytes, filename=model_object_name)
