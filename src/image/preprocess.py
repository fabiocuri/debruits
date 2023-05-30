import glob
import sys
from pathlib import Path

import cv2
from handlers import ImageClass
from numpy import asarray, savez_compressed
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm


def create_model_data(mode, paths):

    """

    Adjusts brightness, contrast and creates canny edges.

    """

    BRIGHTNESS = float(list(sys.argv)[-4])
    CONTRAST = float(list(sys.argv)[-3])
    BLUR = int(list(sys.argv)[-2])
    SATURATION = float(list(sys.argv)[-1])

    for file in tqdm(glob.glob(f"{paths['frames']}/{mode}/*")):

        imagehandler_frame = ImageClass(input_path=file, mode=mode)
        imagehandler_frame.read_image()
        imagehandler_frame.resize((256, 256))
        imagehandler_frame.get_image_name()
        imagehandler_frame.brightness_contrast(BRIGHTNESS, CONTRAST)
        imagehandler_frame.blur(BLUR)
        imagehandler_frame.saturation(SATURATION)
        imagehandler_frame.export_image(output_path=f"{paths['resized']}/{mode}", scale=1)
        
        imagehandler_edges = ImageClass(input_path=file, mode=mode)
        imagehandler_edges.read_image()
        imagehandler_edges.resize((256, 256))
        imagehandler_edges.get_image_name()
        imagehandler_edges.edges_canny()
        imagehandler_edges.export_image(output_path=f"{paths['edges']}/{mode}", scale=1)

        imagehandler_edges.image[:,:,1] = imagehandler_edges.image[:,:,0]
        imagehandler_edges.image[:,:,2] = imagehandler_edges.image[:,:,0]

        concat_img = cv2.hconcat([imagehandler_edges.image, imagehandler_frame.image])

        imagehandler_concat = ImageClass(cv2image=concat_img, mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=file.split("/")[-1].split(".")[0])
        imagehandler_concat.export_image(
            output_path=f"{paths['concat']}/{mode}", scale=1
        )


def load_images(mode, paths):

    """

    Creates train and val zipped data.

    """

    src_list, tar_list = list(), list()

    for file in tqdm(glob.glob(f"{paths['concat']}/{mode}/*")):

        imagehandler_frame = ImageClass(input_path=file)
        imagehandler_frame.read_image()
        pixels = img_to_array(imagehandler_frame.image)

        width = pixels.shape[1]

        edges_img, orig_img = (
            pixels[:, : int(width / 2), :],
            pixels[:, int(width / 2) :, :],
        )
        src_list.append(edges_img)
        tar_list.append(orig_img)

    return [asarray(src_list), asarray(tar_list)]


def preprocess4GAN(mode):

    paths = {
        "frames": "/content/debruits/data/image/input/frames",
        "resized": "/content/debruits/data/image/input/resized",
        "edges": "/content/debruits/data/image/input/edges",
        "concat": "/content/debruits/data/image/input/concat",
    }

    create_model_data(mode=mode, paths=paths)

    [src_images_train, tar_images_train] = load_images(mode=mode, paths=paths)

    Path("/content/drive/MyDrive/image/input/model/").mkdir(parents=True, exist_ok=True)

    savez_compressed(
        f"/content/drive/MyDrive/image/input/model/{mode}.npz",
        src_images_train,
        tar_images_train,
    )


if __name__ == "__main__":

    preprocess4GAN(mode=sys.argv[1])
