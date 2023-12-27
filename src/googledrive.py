import os
from io import BytesIO

import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from matplotlib import pyplot


class GoogleDrive:

    """
    Authenticates to Google Drive storage.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.authenticate()

    def authenticate(self):

        credentials_path = self.config["google_drive"]["service_account_json"]
        scope = ["https://www.googleapis.com/auth/drive"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scope
        )
        self.drive_service = build("drive", "v3", credentials=credentials)

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

    def export_image(self, folder_id, data, idx):

        filename_local = f"step_{idx}.png"

        pyplot.axis("off")
        pyplot.imshow(data)

        pyplot.savefig(filename_local)
        pyplot.close()

        file_metadata = {
            "name": filename_local,
            "parents": [folder_id],
        }

        with open(filename_local, "rb") as file:

            media = MediaIoBaseUpload(file, mimetype="image/png", resumable=True)
            request = self.drive_service.files().create(
                body=file_metadata, media_body=media
            )

            response = None

            while response is None:

                _, response = request.next_chunk()

        os.remove(filename_local)


if __name__ == "__main__":

    GoogleDrive()
