import os
from pathlib import Path

import cv2
import yaml
from PIL import Image
from rembg import remove


class RemoveBackground:

    """
    Description: removes the background of a video.
    Output: video with no background.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.REMOVE_BACKGROUND = self.config["system_config"]["REMOVE_BACKGROUND"]
        self.FPS = self.config["video_config"]["FPS"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]

        self.folder_path = os.path.dirname(self.REMOVE_BACKGROUND)
        self.output_path = f"{self.folder_path}/no_background"

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.remove()

    def sort_key(self, item):
        return int(item.split(".png")[0].split("_")[1])

    def remove(self):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            f"{self.output_path}/no_background.mp4",
            fourcc,
            self.FPS,
            (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT),
        )

        for image in sorted(
            list(os.listdir(self.REMOVE_BACKGROUND)), key=self.sort_key
        ):

            input = Image.open(os.path.join(self.REMOVE_BACKGROUND, image))
            output = remove(input)
            output.save(f"{self.output_path}/{image}")

            frame = cv2.imread(f"{self.output_path}/{image}")
            video_writer.write(frame)

        video_writer.release()


if __name__ == "__main__":

    RemoveBackground()
