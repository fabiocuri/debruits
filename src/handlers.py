import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from skimage.color import label2rgb
from skimage.segmentation import slic


class ImageClass:
    """
    A class that reads, preprocesses and converts images.
    """

    def __init__(self, config, input_path="./", cv2image=None):

        super(ImageClass, self).__init__()
        self.config = config
        self.input_path = input_path
        self.cv2image = cv2image

        self.BLUR = self.config["image_config"]["BLUR"]
        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]

    def read_image(self, flag=cv2.IMREAD_COLOR):

        if self.cv2image is not None:

            self.image = self.cv2image

        else:

            self.image = cv2.imread(self.input_path, flag)

    def resize(self, dim):

        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)

    def get_image_name(self, image_name=None):

        if image_name is not None:

            self.image_name = image_name

        else:

            self.image_name = os.path.basename(self.input_path)

    def apply_filter(self, FILTER):

        if FILTER == "original":

            self.image = self.image

        if "slic" in FILTER:

            n_segments = int(FILTER.split("-")[1])

            self.image_slic = slic(self.image, n_segments=n_segments, compactness=5)
            self.image = label2rgb(self.image_slic, self.image, kind="avg")

        if FILTER == "solarize":

            self.image = Image.fromarray(self.image)
            self.image = ImageOps.solarize(self.image, threshold=130)
            self.image = np.array(self.image)

        if FILTER == "color":

            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        if FILTER == "gaussian":

            self.image = laplace(self.image)

        if FILTER == "edges":

            self.image = cv2.Canny(self.image, 100, 200, 100)
            self.image = np.stack((self.image,) * 3, axis=-1)

        if FILTER == "blur":

            self.image = cv2.blur(self.image, (self.BLUR, self.BLUR))

        if FILTER == "sharpen":

            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.image = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)

    def input_filter(self):

        self.apply_filter(self.INPUT_FILTER)

    def target_filter(self):

        self.apply_filter(self.TARGET_FILTER)

    def export_image(self, output_path, scale):

        Path(output_path).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(f"{output_path}/{self.image_name}.png", scale * self.image)

    def imshow(self):

        cv2.imshow("image", self.image)
        cv2.waitKey(0)
