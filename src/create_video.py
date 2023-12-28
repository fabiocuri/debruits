import os
from io import BytesIO

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from googledrive import GoogleDrive


class Frames2Videos:

    """
    Description: creates a video from image files.
    Output: a .mp4 file.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.drive_service = GoogleDrive()

        self.FPS = self.config["video_config"]["FPS"]
        self.ENHANCED_WIDTH = self.config["image_config"]["ENHANCED_WIDTH"]
        self.ENHANCED_HEIGHT = self.config["image_config"]["ENHANCED_HEIGHT"]
        self.SUPER_RESOLUTION_FOLDER = self.config["system_config"][
            "SUPER_RESOLUTION_FOLDER"
        ]

        self.create_video()

    def create_video(self):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        temp_file_name = "video.mp4"

        video_writer = cv2.VideoWriter(
            temp_file_name,
            fourcc,
            self.FPS,
            (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT),
        )

        for folder in tqdm(self.drive_service.super_resolution_folder_id):

            self.super_resolution_folder_elements_id = self.drive_service.create_folder(
                parent_folder_id=folder, folder_name="super_resolution"
            )

            elements = self.drive_service.get_items_elements(
                self.super_resolution_folder_elements_id
            )
            elements = sorted(
                elements, key=lambda x: int(x["name"].split("_")[1].split(".")[0])
            )

            for element in elements:

                _element = self.drive_service.get_item(element["id"])
                _element = Image.open(BytesIO(_element))
                _element = _element.resize(
                    (self.ENHANCED_WIDTH, self.ENHANCED_HEIGHT), Image.ANTIALIAS
                )
                _element = np.array(_element.convert("RGB"))[:, :, ::-1]
                print(_element.shape)
                video_writer.write(_element)

        video_writer.release()

        with open(temp_file_name, "rb") as video_file:

            video_data = BytesIO(video_file.read())

        video_data.seek(0)

        self.drive_service.send_bytes_file(
            self.drive_service.output_run_id,
            video_data,
            f"{self.SUPER_RESOLUTION_FOLDER}.mp4",
        )

        os.remove(temp_file_name)


if __name__ == "__main__":

    Frames2Videos()
