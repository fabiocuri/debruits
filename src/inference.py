import sys

from tensorflow.keras.models import model_from_json

from mongodb_lib import load_yaml, connect_to_mongodb, preprocess_npz
import numpy as np

class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")
        self.db, self.fs = connect_to_mongodb(config=self.config)

        self.DATASET = sys.argv[1]
        self.INPUT_FILTER = sys.argv[2]
        self.TARGET_FILTER = sys.argv[3]
        self.LEARNING_RATE = sys.argv[4]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

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

        testA, _ = preprocess_npz(fs=self.fs, db=self.db, filename=f"{self.DATASET}_test_preprocessed_{self.model_name}")

        generator_model = self.load_model_from_chunks(
            id_name=f"{self.DATASET}_generator_model_{self.model_name}", db=self.db
        )

        for ix in range(testA.shape[0]):

            X_realA = testA[[ix]]
            X_fakeB = generator_model.predict(X_realA)
            X_fakeB = np.clip(X_fakeB * 255, 0, 255).astype(np.uint8)
            X_fakeB = X_fakeB[0]
            image_bytes = X_fakeB.astype(np.uint8).tobytes()
            filename = f"{self.DATASET}_test_inference_{ix}_step_0_{self.model_name}"
            self.fs.put(image_bytes, filename=filename)


if __name__ == "__main__":

    Inference()
