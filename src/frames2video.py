import os
import sys

import cv2
import yaml


def frames2videos():

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)

    FRAMES_FOLDER = list(sys.argv)[-1]

    folder = config["folder"]

    FPS = config["frames_to_video_config"]["FPS"]

    INPUT_FILTER = config["model_config"]["INPUT_FILTER"]
    TARGET_FILTER = config["model_config"]["TARGET_FILTER"]

    enhanced_width = config["enhanced_width"]
    enhanced_height = config["enhanced_height"]

    model_config = f"{INPUT_FILTER}_{TARGET_FILTER}"

    plot_folders = sorted(os.listdir(f"./data/{folder}/image/output/{model_config}"))

    if FRAMES_FOLDER == "plots":

        plot_folders = [
            pf
            for pf in plot_folders
            if pf.startswith("plot_") and not pf.endswith(".mp4")
        ]

    if FRAMES_FOLDER == "inference":

        plot_folders = [
            pf
            for pf in plot_folders
            if pf.startswith("inference") and not pf.endswith(".mp4")
        ]

    base_path = f"./data/{folder}/image/output/{model_config}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"{base_path}/{FRAMES_FOLDER}.mp4", fourcc, FPS, (enhanced_width, enhanced_height)
    )

    for plot_folder in plot_folders:

        path = f"{base_path}/{plot_folder}"

        if FRAMES_FOLDER == "plots":

            def sort_key(item):
                return int(item.split("_")[1].split(".png")[0])

            plot_files = sorted(
                list(os.listdir(path)),
                key=sort_key,
            )

        if FRAMES_FOLDER == "inference":

            def sort_key(item):
                return int(item.split(".png")[0])

            plot_files = sorted(
                list(os.listdir(path)),
                key=sort_key,
            )

        plot_files = [pf for pf in plot_files if pf.endswith(".png")]

        for pf in plot_files:
            frame_path = os.path.join(path, pf)
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":

    frames2videos()
