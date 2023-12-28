from io import BytesIO

import yaml
from PIL import Image, ImageEnhance
from tqdm import tqdm

from googledrive import GoogleDrive


class SuperResolution:

    """
    Description: improves the quality of the infered images.
    Output: improved images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.drive_service = GoogleDrive()

        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.ENHANCEMENT_FACTOR = self.config["image_config"]["ENHANCEMENT_FACTOR"]

        self.improve()

    def improve(self):

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
