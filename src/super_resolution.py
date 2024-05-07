from io import BytesIO

import yaml
from gridfs import GridFS
from PIL import Image, ImageEnhance
from tqdm import tqdm

from encode_images import (
    connect_to_mongodb,
    delete_all_documents_in_collection,
    load_yaml,
)


class SuperResolution:

    """
    Description: improves the quality of the infered images.
    Output: improved images.
    """

    def __init__(self):

        self.yaml_data = load_yaml("./debruits-kubernetes/values.yaml")
        self.db = connect_to_mongodb(self.yaml_data)

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        LEARNING_RATE = self.config["model_config"]["LEARNING_RATE"]

        self.model_name = f"inputfilter_{INPUT_FILTER}_targetfilter_{TARGET_FILTER}_lr_{LEARNING_RATE}"

        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.ENHANCEMENT_FACTOR = self.config["image_config"]["ENHANCEMENT_FACTOR"]

        self.collection_test_inference_super_resolution = (
            self.yaml_data[f"mongoDbtestCollectionInferenceSuperResolution"]
            + "_"
            + self.model_name
        )
        delete_all_documents_in_collection(
            self.db, self.collection_test_inference_super_resolution
        )
        self.collection_test_inference_super_resolution = self.db[
            self.collection_test_inference_super_resolution
        ]

        self.collection_test_evolution_super_resolution = (
            self.yaml_data[f"mongoDbtestCollectionEvolutionSuperResolution"]
            + "_"
            + self.model_name
        )
        delete_all_documents_in_collection(
            self.db, self.collection_test_evolution_super_resolution
        )
        self.collection_test_evolution_super_resolution = self.db[
            self.collection_test_evolution_super_resolution
        ]

        self.fs = GridFS(self.db)

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
