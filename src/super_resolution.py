from io import BytesIO

from PIL import Image, ImageEnhance
from tqdm import tqdm

from encode_images import (
    connect_to_mongodb,
    load_yaml,
)
import sys

class SuperResolution:

    """
    Description: improves the quality of the infered images.
    Output: improved images.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")
        self.db, self.fs = connect_to_mongodb(config = self.config)

        self.INPUT_FILTER = sys.argv[1]
        self.TARGET_FILTER = sys.argv[2]
        self.LEARNING_RATE = sys.argv[3]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.ENHANCEMENT_FACTOR = self.config["image_config"]["ENHANCEMENT_FACTOR"]

        self.improve()

    def improve(self):

        file = self.fs.find_one({"filename": "image_1_final"})

        chunks_cursor = self.db.fs.chunks.find({"files_id": file._id}).sort("n", 1)
        model_json_bytes = b"".join(chunk["data"] for chunk in chunks_cursor)
        data = model_json_bytes.decode()

        print(data.shape)

        import sys

        sys.exit()

        for folder in tqdm(self.drive_service.super_resolution_folder_id):

            self.super_resolution_folder_elements_id = self.drive_service.create_folder(
                parent_folder_id=folder, folder_name="super_resolution"
            )

            elements = self.drive_service.get_items_elements(folder)

            for element in elements:

                if element["name"] != "super_resolution":

                    _element = self.drive_service.get_item(element["id"])
                    _element = Image.open(BytesIO(_element))

                    resized_element = _element.resize(
                        (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT), Image.ANTIALIAS
                    )

                    enhancer = ImageEnhance.Sharpness(resized_element)
                    improved_element = enhancer.enhance(self.ENHANCEMENT_FACTOR)

                    self.drive_service.export_image(
                        folder_id=self.super_resolution_folder_elements_id,
                        data=improved_element,
                        file_name=element["name"],
                    )


if __name__ == "__main__":

    SuperResolution()
