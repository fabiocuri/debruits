import base64
import os

import yaml
from pymongo import MongoClient
from tqdm import tqdm


def Encode():

    with open("./debruits-kubernetes/values.yaml", "r") as file:

        yaml_data = yaml.safe_load(file)

    mongodbUsername = base64.b64decode(str(yaml_data["mongodbUsernameBase64"])).decode(
        "utf-8"
    )
    mongodbPassword = base64.b64decode(str(yaml_data["mongodbPasswordBase64"])).decode(
        "utf-8"
    )
    mongoDbPort = str(yaml_data["mongoDbPort"])

    client = MongoClient(
        f"mongodb://{mongodbUsername}:{mongodbPassword}@localhost:{mongoDbPort}/?authSource=admin"
    )

    db = client[yaml_data["mongoDbDatabase"]]

    for data_type in ["train", "test"]:

        collection = db[yaml_data[f"mongoDb{data_type}Collection"]]

        images_dir = f"./data/{data_type}"

        for filename in tqdm(os.listdir(images_dir)):

            if filename.endswith(".JPG") or filename.endswith(".jpg"):

                image_path = os.path.join(images_dir, filename)

                with open(image_path, "rb") as image_file:

                    image_data = image_file.read()

                base64_image = base64.b64encode(image_data)

                base64_image_str = base64_image.decode("utf-8")

                image_doc = {"filename": filename, "base64_image": base64_image_str}

                collection.insert_one(image_doc)

        print("Images encoded and saved to MongoDB successfully.")


if __name__ == "__main__":

    Encode()
