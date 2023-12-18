from pathlib import Path

import yaml
from tensorflow.keras.models import load_model
from tqdm import tqdm

from handlers import ImageClass
from train import load_compressed


class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.data = self.config["data"]
        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]

        self.models_path = (
            f"./data/{self.data}/image/output/{self.INPUT_FILTER}_{self.TARGET_FILTER}"
        )

        self.infere()

    def infere(self):

        Path(f"{self.models_path}/inference/").mkdir(parents=True, exist_ok=True)

        trainA, trainB = load_compressed(self.data, "test.npz")

        generator_model = load_model(
            f"{self.models_path}/trained_models/generator_model.h5"
        )

        for ix in tqdm(range(0, trainA.shape[0], 1)):

            X1, _ = trainA[[ix]], trainB[[ix]]

            X_fakeB = generator_model.predict(X1)

            X_fakeB = (X_fakeB + 1) / 2.0

            imagehandler_concat = ImageClass(config=self.config, cv2image=X_fakeB[0])
            imagehandler_concat.read_image()
            imagehandler_concat.get_image_name(image_name=f"image_{ix}")
            imagehandler_concat.export_image(
                output_path=f"{self.models_path}/inference", scale=255
            )


if __name__ == "__main__":

    Inference()
