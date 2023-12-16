import yaml
import glob
import cv2
from pathlib import Path
import random
import numpy as np

if __name__ == "__main__":

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)
    folder = config["folder"]

    overlap_folders = config["overlap-results"]

    inference_plots = glob.glob(f"./data/{config['folder']}/image/output/{overlap_folders[0]}/inference/*")
    inference_plots = [file.split("/")[-1] for file in inference_plots]

    output_path = f"./data/{config['folder']}/image/output/overlap-{overlap_folders[0]}-{overlap_folders[1]}"

    for file in inference_plots:

        file = random.choice(inference_plots)

        image_0 = cv2.imread(f"./data/{config['folder']}/image/output/{overlap_folders[0]}/inference/{file}")

        file = random.choice(inference_plots)

        image_1 = cv2.imread(f"./data/{config['folder']}/image/output/{overlap_folders[1]}/inference/{file}")

        overlap = cv2.addWeighted(image_0, 1, image_1, 1, 0)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_path}/{file}", overlap)