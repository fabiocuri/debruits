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

        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]

        self.client_googledrive = GoogleDrive()

        # The main_id is the ID of the name of the dataset in Google Drive,
        # i.e. the ID of "data/bairrodorego".
        self.main_id = self.config["google_drive"]["main_id"]

        self.input_id = self.client_googledrive.create_folder(
            parent_folder_id=self.main_id, folder_name="input"
        )

        self.train_folder_id = self.client_googledrive.create_folder(
            parent_folder_id=self.input_id, folder_name="train"
        )

        self.test_folder_id = self.client_googledrive.create_folder(
            parent_folder_id=self.input_id, folder_name="test"
        )

        self.model_data_id = self.client_googledrive.create_folder(
            parent_folder_id=self.main_id, folder_name="model_data"
        )

        self.model_data_run_id = self.client_googledrive.create_folder(
            parent_folder_id=self.model_data_id,
            folder_name=f"{self.INPUT_FILTER}_{self.TARGET_FILTER}",
        )

        self.preprocess_data(mode="train", folder_id=self.train_folder_id)
        self.preprocess_data(mode="test", folder_id=self.test_folder_id)

    def preprocess_data(self, mode, folder_id):

        src_list, tar_list = [], []

        ids = self.client_googledrive.get_items_elements(folder_id=folder_id)

        for index, file in enumerate(tqdm(ids)):

            input_img = ImageClass(
                image_element=file, client_googledrive=self.client_googledrive
            )
            input_img.input_filter()

            target_img = ImageClass(
                image_element=file, client_googledrive=self.client_googledrive
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
        self.client_googledrive.send_bytes_file(
            folder_id=self.model_data_run_id, bytes_io=npz_data, file_name=f"{mode}.npz"
        )


if __name__ == "__main__":

    Preprocess()
