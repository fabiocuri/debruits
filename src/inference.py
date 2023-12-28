import yaml
from tqdm import tqdm

from googledrive import GoogleDrive
from train import load_npz


class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.drive_service = GoogleDrive()

        self.infere()

    def infere(self):

        trainA, trainB = load_npz(drive_service=self.drive_service, dataset="test")

        item_id = self.drive_service.get_item_id_by_name(
            self.drive_service.trained_models_run_id, "generator_model.h5"
        )

        generator_model = self.drive_service.read_h5_file(item_id)

        for ix in tqdm(range(0, trainA.shape[0], 1)):

            X1, _ = trainA[[ix]], trainB[[ix]]

            X_fakeB = generator_model.predict(X1)

            X_fakeB = (X_fakeB + 1) / 2.0

            X_fakeB = X_fakeB[0]

            self.drive_service.export_image(
                folder_id=self.drive_service.inference_folder_id,
                data=X_fakeB,
                file_name=f"step_{ix}.png",
            )


if __name__ == "__main__":

    Inference()
