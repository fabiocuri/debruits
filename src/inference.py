from tensorflow.keras.models import model_from_json
from tqdm import tqdm

from encode_images import connect_to_mongodb, load_yaml
from train import preprocess_chunks


class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")
        self.db, self.fs = connect_to_mongodb(config=self.config)

        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.LEARNING_RATE = self.config["model_config"]["LEARNING_RATE"]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        self.infere()

    def load_model_from_chunks(self, id_name, db):

        file = self.fs.find_one({"filename": id_name})

        chunks_cursor = db.fs.chunks.find({"files_id": file._id}).sort("n", 1)
        model_json_bytes = b"".join(chunk["data"] for chunk in chunks_cursor)
        model_json = model_json_bytes.decode()
        model = model_from_json(model_json)

        return model

    def infere(self):

        trainA, trainB = preprocess_chunks(
            fs=self.fs, id_name=f"test_preprocessed_{self.model_name}", db=self.db
        )
        generator_model = self.load_model_from_chunks(
            id_name=f"generator_model_{self.model_name}", db=self.db
        )

        for ix in tqdm(range(0, trainA.shape[0], 1)):

            X1, _ = trainA[[ix]], trainB[[ix]]

            X_fakeB = generator_model.predict(X1)

            X_fakeB = (X_fakeB + 1) / 2.0

            X_fakeB = X_fakeB[0]

            image_bytes = X_fakeB.tobytes()
            filename = f"test_image_{ix}_{self.model_name}_inference"
            self.fs.put(image_bytes, filename=filename)


if __name__ == "__main__":

    Inference()
