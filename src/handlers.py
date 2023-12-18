import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from skimage.color import label2rgb
from skimage.segmentation import slic

class VideoClass:
    """
    A class that reads, preprocesses and converts videos.
    """

    def __init__(self, input_path="./"):

        super(VideoClass, self).__init__()
        self.input_path = input_path

    def read_video(self):

        self.video = cv2.VideoCapture(self.input_path)

    def get_video_length(self):

        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        return duration % 60

    def get_video_name(self):

        base = os.path.basename(self.input_path)
        self.video_name = os.path.splitext(base)[0]

    def resize_image(self, image, dim):

        return cv2.resize(image, dim)

    def video2frames(self, output_path, dim):

        output_path = f"{output_path}/{self.video_name}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        success, image = self.video.read()
        dim = (image.shape[0], image.shape[1])
        image = self.resize_image(image, dim)
        count = 1

        while success:

            cv2.imwrite(f"{output_path}/frame_{count}.jpg", image)

            success, image = self.video.read()

            try:

                image = self.resize_image(image, dim)
                count += 1

            except:

                pass

    def imshow(self):

        while self.video.isOpened():

            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame", gray)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

        self.video.release()
        cv2.destroyAllWindows()


class ImageClass:
    """
    A class that reads, preprocesses and converts images.
    """

    def __init__(self, config, input_path="./", mode="train", cv2image=None):

        super(ImageClass, self).__init__()
        self.config = config
        self.input_path = input_path
        self.mode = mode
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

        if FILTER == "slic-100":

            self.image_slic = slic(self.image, n_segments=100, compactness=5)
            self.image = label2rgb(self.image_slic, self.image, kind="avg")

        if FILTER == "slic-500":

            self.image_slic = slic(self.image, n_segments=500, compactness=5)
            self.image = label2rgb(self.image_slic, self.image, kind="avg")

        if FILTER == "slic-1000":

            self.image_slic = slic(self.image, n_segments=1000, compactness=5)
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
