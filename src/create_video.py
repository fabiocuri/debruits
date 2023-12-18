import os
import sys

import cv2
import yaml
from tqdm import tqdm


class Frames2Videos:

    """
    Description: creates a video from image files.
    Output: a .mp4 file.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.SCRIPT_FOLDER = list(sys.argv)[-1]

        self.data = self.config["data"]
        self.FPS = self.config["video_config"]["FPS"]
        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]

        self.models_path = (
            f"./data/{self.data}/image/output/{self.INPUT_FILTER}_{self.TARGET_FILTER}"
        )

        self.create_video()

    def sort_key(self, item):
        return int(item.split(".png")[0].split("_")[1])

    def create_video(self):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            f"{self.models_path}/{self.SCRIPT_FOLDER}.mp4",
            fourcc,
            self.FPS,
            (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT),
        )

        for plot_folder in tqdm(
            [
                os.path.join(self.models_path, folder)
                for folder in os.listdir(self.models_path)
                if os.path.isdir(os.path.join(self.models_path, folder))
                and folder.startswith(self.FOLDER)
            ]
        ):

            for pf in sorted(list(os.listdir(plot_folder)), key=self.sort_key):

                frame = cv2.imread(os.path.join(plot_folder, pf))
                video_writer.write(frame)

        video_writer.release()


if __name__ == "__main__":

    Frames2Videos()
