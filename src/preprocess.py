from io import BytesIO

import cv2
import yaml
from numpy import asarray, savez_compressed
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

from googledrive import GoogleDrive
from handlers import ImageClass


class Preprocess:

    """
    Description: preprocessing of train and test images.
    Output: .npz files with packed preprocessed images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.drive_service = GoogleDrive()

        self.preprocess_data(mode="train", folder_id=self.drive_service.train_folder_id)
        self.preprocess_data(mode="test", folder_id=self.drive_service.test_folder_id)

    def preprocess_data(self, mode, folder_id):

        src_list, tar_list = [], []

        ids = self.drive_service.get_items_elements(folder_id=folder_id)

        ids = ids[:3]

        for index, file in enumerate(tqdm(ids)):

            input_img = ImageClass(image_element=file, drive_service=self.drive_service)
            input_img.input_filter()

            target_img = ImageClass(
                image_element=file, drive_service=self.drive_service
            )
            target_img.target_filter()

            src_list.append(img_to_array(input_img.image))
            tar_list.append(img_to_array(input_img.image))

            if mode == "train" and index == 0:

                concat_img = cv2.hconcat([input_img.image, target_img.image])

                cv2.imshow("Image", concat_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        [src_images_train, tar_images_train] = [asarray(src_list), asarray(tar_list)]

        npz_data = BytesIO()
        savez_compressed(npz_data, src_images_train, tar_images_train)
        self.drive_service.send_bytes_file(
            folder_id=self.drive_service.model_data_run_id,
            bytes_io=npz_data,
            file_name=f"{mode}.npz",
        )


if __name__ == "__main__":

    Preprocess()
