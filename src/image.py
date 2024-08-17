import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from skimage.color import label2rgb
from skimage.segmentation import slic


class ImageClass:
    """
    A class that reads and preprocesses images.
    """

    def __init__(self, image):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.image = image

        self.IMAGE_DIM = self.config["image_config"]["DIM"]
        self.BLUR = self.config["image_config"]["BLUR"]

        self.resize()

    def resize(self):

        self.image = cv2.resize(
            self.image, (self.IMAGE_DIM, self.IMAGE_DIM), interpolation=cv2.INTER_LINEAR
        )

    def apply_filter(self, FILTER):

        if FILTER == "original":

            self.image = self.image

        if "slic" in FILTER:

            n_segments = int(FILTER.split("-")[1])

            self.image_slic = slic(self.image, n_segments=n_segments, compactness=1)
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

        if FILTER == "special-input":

            self.image_slic = slic(self.image, n_segments=1000, compactness=1)
            self.image = label2rgb(self.image_slic, self.image, kind="avg")

            self.image = laplace(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

            self.image = cv2.blur(self.image, (50, 50))

        if FILTER == "special-target":

            self.image_slic = slic(self.image, n_segments=1000, compactness=1)
            self.image = label2rgb(self.image_slic, self.image, kind="avg")

            self.image = laplace(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def input_filter(self, INPUT_FILTER):

        self.apply_filter(INPUT_FILTER)

    def target_filter(self, TARGET_FILTER):

        self.apply_filter(TARGET_FILTER)
