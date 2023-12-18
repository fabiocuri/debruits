from rembg import remove 
from PIL import Image 
import glob
import os
import sys
from tqdm import tqdm
import cv2
import yaml

def main(path):

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)

    video_name = path.split("/")[-1]

    images = [f for f in glob.glob(f"{path}/*.jpg")]
    images = list(sorted(images))

    output_path = f"{path}/no_background"

    if not os.path.exists(output_path):

        os.makedirs(output_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    FPS = config["frames_to_video_config"]["FPS"]

    enhanced_width = config["enhanced_width"]
    enhanced_height = config["enhanced_height"]

    video_writer = cv2.VideoWriter(
        f"{output_path}/{video_name}.mp4",
        fourcc,
        FPS,
        (enhanced_width, enhanced_height),
    )

    for image in tqdm(images):

        file_name = image.split("/")[-1].replace("jpg", "png")
        
        input = Image.open(image)
        output = remove(input) 
        output.save(f"{output_path}/{file_name}")

        frame = cv2.imread(f"{output_path}/{file_name}")
        video_writer.write(frame)

    video_writer.release()

if __name__ == "__main__":

    main(sys.argv[-1])
