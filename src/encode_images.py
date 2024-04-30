import base64
import os

from pymongo import MongoClient
from tqdm import tqdm

uri = "mongodb://debruits:debruits@localhost:27017/?authSource=admin"

client = MongoClient(uri)
db = client.debruits
collection = db.train

images_dir = "data/train"

for filename in tqdm(os.listdir(images_dir)):
    if (
        filename.endswith(".jpg")
        or filename.endswith(".jpeg")
        or filename.endswith(".png")
    ):

        image_path = os.path.join(images_dir, filename)

        with open(image_path, "rb") as image_file:

            image_data = image_file.read()

        base64_image = base64.b64encode(image_data)

        base64_image_str = base64_image.decode("utf-8")

        image_doc = {"filename": filename, "base64_image": base64_image_str}

        collection.insert_one(image_doc)

print("Images encoded and saved to MongoDB successfully.")
