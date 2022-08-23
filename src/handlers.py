import os
from pathlib import Path

import cv2
import numpy as np


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

    def __init__(self, input_path="./", mode="train", cv2image=None):

        super(ImageClass, self).__init__()
        self.input_path = input_path
        self.mode = mode
        self.cv2image = cv2image

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

    def brightness_contrast(self, BRIGHTNESS, CONTRAST):

        new_image = self.image.copy()

        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                for c in range(self.image.shape[2]):
                    new_image[y, x, c] = np.clip(
                        CONTRAST * self.image[y, x, c] + BRIGHTNESS, 0, 255
                    )

        self.image = new_image

    def blur(self, BLUR):

        self.image = cv2.blur(self.image, (BLUR, BLUR))

    def saturation(self, SATURATION):

        (h, s, v) = cv2.split(self.image)
        s = s * SATURATION
        s = np.clip(s, 0, 255)
        self.image = cv2.merge([h, s, v])

        self.image = cv2.cvtColor(self.image.astype("uint8"), cv2.COLOR_HSV2BGR)

    def edges_canny(self):

        self.image = cv2.Canny(self.image, 100, 200)

    def export_image(self, output_path):

        Path(output_path).mkdir(parents=True, exist_ok=True)

        print("----------------------------")
        print(f"{output_path}/{self.image_name}")

        cv2.imwrite(f"{output_path}/{self.image_name}.jpg", 255 * self.image)

    def imshow(self):

        cv2.imshow("image", self.image)
        cv2.waitKey(0)
