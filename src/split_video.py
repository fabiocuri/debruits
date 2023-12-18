import os
from pathlib import Path

import cv2
import yaml


class SplitVideo:

    """
    Description: splits a video into several frames.
    Output: frames.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.VIDEO_PATH = self.config["system_config"]["VIDEO_PATH"]
        self.FPS = self.config["video_config"]["FPS"]

        self.folder_path = os.path.dirname(self.VIDEO_PATH)
        self.output_path = f"{self.folder_path}/frames"

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.extract_frames()

    def extract_frames(self):

        video_capture = cv2.VideoCapture(self.VIDEO_PATH)
        frames_per_second = int(video_capture.get(self.FPS))
        frame_interval = int(frames_per_second / self.FPS)

        frame_count = 0
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = (
                    f"{self.output_path}/frame_{frame_count // frame_interval}.png"
                )
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        video_capture.release()


if __name__ == "__main__":

    SplitVideo()
