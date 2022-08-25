import glob
import sys
from pathlib import Path
import cv2
from numpy import asarray, savez_compressed
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

from handlers import ImageClass


def effects_and_canny(mode, paths):

    """

    Adjusts brightness, contrast and creates canny edges.

    """

    BRIGHTNESS = int(list(sys.argv)[-4])
    CONTRAST = int(list(sys.argv)[-3])
    BLUR = int(list(sys.argv)[-2])
    SATURATION = int(list(sys.argv)[-1])

    for file in tqdm(glob.glob(f"{paths['frames']}/{mode}/*")):

        imagehandler = ImageClass(input_path=file, mode=mode)
        imagehandler.read_image()
        imagehandler.resize((256, 256))
        imagehandler.get_image_name()
        imagehandler.brightness_contrast(BRIGHTNESS, CONTRAST)
        imagehandler.blur(BLUR)
        imagehandler.saturation(SATURATION)
        imagehandler.export_image(output_path=f"{paths['resized']}/{mode}", scale=255)
        imagehandler.edges_canny()
        imagehandler.export_image(output_path=f"{paths['edges']}/{mode}", scale=1)


def concat2model(mode, paths):

    """

    Concatenates source and target images.

    """

    for file in tqdm(glob.glob(f"{paths['edges']}/{mode}/*")):

        image_name = file.split("/")[-1]

        imagehandler_frame = ImageClass(
            input_path=f"{paths['resized']}/{mode}/{image_name}", mode=mode
        )
        imagehandler_frame.read_image()

        imagehandler_edges = ImageClass(input_path=file, mode=mode)
        imagehandler_edges.read_image()

        concat_img = cv2.hconcat([imagehandler_edges.image, imagehandler_frame.image])

        imagehandler_concat = ImageClass(cv2image=concat_img, mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=image_name)
        imagehandler_concat.export_image(
            output_path=f"{paths['concat']}/{mode}", scale=255
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
        "frames": "/content/debruits/data/input/frames",
        "resized": "/content/debruits/data/input/resized",
        "edges": "/content/debruits/data/input/edges",
        "concat": "/content/debruits/data/input/concat",
    }

    effects_and_canny(mode=mode, paths=paths)

    concat2model(mode=mode, paths=paths)

    [src_images_train, tar_images_train] = load_images(mode=mode, paths=paths)
    
    Path("/content/drive/MyDrive/input/model/").mkdir(parents=True, exist_ok=True)

    savez_compressed("/content/drive/MyDrive/input/model/{mode}.npz", src_images_train, tar_images_train)


if __name__ == "__main__":

    preprocess4GAN(mode=sys.argv[1])
