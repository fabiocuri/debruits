import os
import sys

import yaml
from PIL import Image, ImageEnhance
from tqdm import tqdm

class SuperResolution:

    """
    Description: improves the quality of the inferred images.
    Output: improved images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.FOLDER = list(sys.argv)[-1]

        self.data = self.config["data"]
        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]

        self.models_path = (
            f"./data/{self.data}/image/output/{self.INPUT_FILTER}_{self.TARGET_FILTER}"
        )

        self.improve()

    def improve(self):

        for plot_folder in tqdm([os.path.join(self.models_path, folder) for folder in os.listdir(self.models_path) if os.path.isdir(os.path.join(self.models_path, folder)) and folder.startswith(self.FOLDER)]):

            for pf in os.listdir(plot_folder):

                image_path = os.path.join(plot_folder, pf)

                input_image = Image.open(image_path)

                resized_image = input_image.resize(
                    (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT), Image.ANTIALIAS
                )

                enhancer = ImageEnhance.Sharpness(resized_image)
                quality_factor = 20.0
                improved_quality_image = enhancer.enhance(quality_factor)

                improved_quality_image.save(image_path)

if __name__ == "__main__":

    SuperResolution()
