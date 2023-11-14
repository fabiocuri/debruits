import os
import sys

import yaml
from PIL import Image, ImageEnhance
from tqdm import tqdm

if __name__ == "__main__":

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)

    SUPER_RESOLUTION_FOLDER = list(sys.argv)[-1]

    folder = config["folder"]
    enhanced_width = config["enhanced_width"]
    enhanced_height = config["enhanced_height"]

    INPUT_FILTER = config["model_config"]["INPUT_FILTER"]
    TARGET_FILTER = config["model_config"]["TARGET_FILTER"]

    model_config = f"{INPUT_FILTER}_{TARGET_FILTER}"

    # Get the list of plot folders

    plot_folders = sorted(os.listdir(f"./data/{folder}/image/output/{model_config}"))

    if SUPER_RESOLUTION_FOLDER == "plots":

        plot_folders = [
            pf
            for pf in plot_folders
            if pf.startswith("plot_") and not pf.endswith(".mp4")
        ]

    if SUPER_RESOLUTION_FOLDER == "inference":

        plot_folders = [
            pf
            for pf in plot_folders
            if pf.startswith("inference") and not pf.endswith(".mp4")
        ]

    for plot_folder in tqdm(plot_folders):

        path = f"./data/{folder}/image/output/{model_config}/{plot_folder}"

        plot_files = sorted(os.listdir(path))

        plot_files = [pf for pf in plot_files if pf.endswith(".png")]

        for pf in plot_files:

            frame_path = os.path.join(path, pf)

            input_image = Image.open(frame_path)

            resized_image = input_image.resize(
                (enhanced_width, enhanced_height), Image.ANTIALIAS
            )

            enhancer = ImageEnhance.Sharpness(resized_image)
            quality_factor = 20.0
            improved_quality_image = enhancer.enhance(quality_factor)

            improved_quality_image.save(frame_path)
