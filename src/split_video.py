import os
import sys

import cv2


def extract_frames(input_video_path, frame_rate=5):
    video_capture = cv2.VideoCapture(input_video_path)
    frames_per_second = int(video_capture.get(5))  # Get frames per second of the video

    # Calculate the frame interval based on the desired frame rate
    frame_interval = int(frames_per_second / frame_rate)

    output_folder_path = "/".join(input_video_path.split("/")[:-1])

    video_name = input_video_path.split("/")[-1].replace(".mpg", "")

    if not os.path.exists(f"{output_folder_path}/{video_name}"):

        os.makedirs(f"{output_folder_path}/{video_name}")

    frame_count = 0
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Save frame if it's a multiple of frame_interval
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder_path}/{video_name}/frame_{frame_count // frame_interval}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    video_capture.release()


if __name__ == "__main__":

    extract_frames(sys.argv[-1])
