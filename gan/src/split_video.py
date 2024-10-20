import os
import sys

import cv2
import yaml
import uuid

class SplitVideo:

    """
    Description: splits a video into several frames.
    Output: frames.
    """

    def __init__(self):

        self.config = yaml.load(open("config_pipeline.yaml"), Loader=yaml.FullLoader)

        self.VIDEO_PATH = sys.argv[1]

        self.FPS = int(self.config["video_config"]["FPS"])

        self.output_path = "frames"
        os.makedirs(self.output_path, exist_ok=True)

        self.extract_frames()

    def extract_frames(self):

        video_capture = cv2.VideoCapture(self.VIDEO_PATH)
        frames_per_second = int(video_capture.get(self.FPS))
        frame_interval = int(frames_per_second / self.FPS)

        frame_count = 0

        uid=str(uuid.uuid4())

        while True:

            ret, frame = video_capture.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = (
                    f"{self.output_path}/{uid}_frame_{frame_count // frame_interval}.png"
                )
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        video_capture.release()


if __name__ == "__main__":

    SplitVideo()
