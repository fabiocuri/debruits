import os
from io import BytesIO

import h5py
import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from keras.models import load_model
from matplotlib import pyplot


class GoogleDrive:

    """
    Authenticates to Google Drive storage.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.SUPER_RESOLUTION_FOLDER = self.config["system_config"][
            "SUPER_RESOLUTION_FOLDER"
        ]

        self.authenticate()
        self.handle_folders()

    def authenticate(self):

        credentials_path = self.config["google_drive"]["service_account_json"]
        scope = ["https://www.googleapis.com/auth/drive"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scope
        )
        self.drive_service = build("drive", "v3", credentials=credentials)

    def handle_folders(self):

        run = f"{self.INPUT_FILTER}_{self.TARGET_FILTER}"

        # The main_id is the ID of the name of the dataset in Google Drive,
        # i.e. the ID of "data/bairrodorego".
        self.main_id = self.config["google_drive"]["main_id"]

        self.input_id = self.create_folder(
            parent_folder_id=self.main_id, folder_name="input"
        )

        self.train_folder_id = self.create_folder(
            parent_folder_id=self.input_id, folder_name="train"
        )

        self.test_folder_id = self.create_folder(
            parent_folder_id=self.input_id, folder_name="test"
        )

        self.model_data_id = self.create_folder(
            parent_folder_id=self.main_id, folder_name="model_data"
        )

        self.model_data_run_id = self.create_folder(
            parent_folder_id=self.model_data_id, folder_name=run
        )

        self.trained_models_id = self.create_folder(
            parent_folder_id=self.main_id, folder_name="trained_models"
        )

        self.trained_models_run_id = self.create_folder(
            parent_folder_id=self.trained_models_id, folder_name=run
        )

        self.output_id = self.create_folder(
            parent_folder_id=self.main_id, folder_name="output"
        )

        self.output_run_id = self.create_folder(
            parent_folder_id=self.output_id, folder_name=run
        )

        self.inference_folder_id = self.create_folder(
            parent_folder_id=self.output_run_id, folder_name="inference"
        )

        if self.SUPER_RESOLUTION_FOLDER == "inference":

            self.super_resolution_folder_id = [self.inference_folder_id]

        elif self.SUPER_RESOLUTION_FOLDER == "evolution":

            self.evolution_folders = self.get_items_elements(self.output_run_id)

            self.super_resolution_folder_id = [
                element["id"]
                for element in self.evolution_folders
                if element["name"].startswith("evolution")
            ]

    def create_folder(self, parent_folder_id, folder_name):

        existing_folder_id = self.get_item_id_by_name(parent_folder_id, folder_name)

        if not existing_folder_id:

            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_folder_id],
            }

            folder = (
                self.drive_service.files()
                .create(body=file_metadata, fields="id")
                .execute()
            )

            return folder.get("id")

        else:

            return existing_folder_id

    def get_items_elements(self, folder_id):

        results = (
            self.drive_service.files().list(q=f"'{folder_id}' in parents").execute()
        )

        return results.get("files", [])

    def send_bytes_file(self, folder_id, bytes_io, file_name):

        file_metadata = {
            "name": file_name,
            "parents": [folder_id],
        }

        media = MediaIoBaseUpload(
            bytes_io, mimetype="application/octet-stream", resumable=True
        )

        existing_folder_id = self.get_item_id_by_name(folder_id, file_name)

        if not existing_folder_id:

            request = self.drive_service.files().create(
                body=file_metadata, media_body=media
            )

        else:

            file_metadata = {"addParents": existing_folder_id}

            request = self.drive_service.files().update(
                fileId=existing_folder_id, body=file_metadata, media_body=media
            )

        response = None

        while response is None:

            _, response = request.next_chunk()

        return response

    def get_item_id_by_name(self, folder_id, file_name):

        results = (
            self.drive_service.files()
            .list(q=f"'{folder_id}' in parents and name = '{file_name}'")
            .execute()
        )

        files = results.get("files", [])

        if files:

            return files[0]["id"]

        else:

            return None

    def get_item(self, item_id):

        request = self.drive_service.files().get_media(fileId=item_id)
        image_data = BytesIO()
        downloader = MediaIoBaseDownload(image_data, request)
        done = False

        while not done:

            _, done = downloader.next_chunk()

        return image_data.getvalue()

    def export_image(self, folder_id, data, file_name):

        pyplot.axis("off")
        pyplot.imshow(data)

        pyplot.savefig(file_name)
        pyplot.close()

        file_metadata = {
            "name": file_name,
            "parents": [folder_id],
        }

        with open(file_name, "rb") as file:

            media = MediaIoBaseUpload(file, mimetype="image/png", resumable=True)

            request = self.drive_service.files().create(
                body=file_metadata, media_body=media
            )

            response = None

            while response is None:

                _, response = request.next_chunk()

        os.remove(file_name)

    def export_h5_file(self, folder_id, model, file_name):

        model.save("temp.h5")

        with open("temp.h5", "rb") as f:
            h5_data = BytesIO(f.read())

        os.remove("temp.h5")

        h5_data.seek(0)

        self.send_bytes_file(folder_id, h5_data, file_name)

    def read_h5_file(self, item_id):

        file_data = self.get_item(item_id)
        h5_file = h5py.File(BytesIO(file_data), "r")

        model = load_model(h5_file)

        return model


if __name__ == "__main__":

    GoogleDrive()
