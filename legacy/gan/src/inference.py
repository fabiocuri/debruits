import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json

from mongodb_lib import (
    connect_to_mongodb,
    load_yaml,
    preprocess_npz,
    preprocess_npz_local,
)


class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.config = load_yaml("config_pipeline.yaml")

        self.MODE = sys.argv[1]
        self.DATASET = sys.argv[2]
        self.INPUT_FILTER = sys.argv[3]
        self.TARGET_FILTER = sys.argv[4]
        self.LEARNING_RATE = sys.argv[5]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        if self.MODE == "jenkins":

            self.db, self.fs = connect_to_mongodb(config=self.config)
            self.testA, _ = preprocess_npz(
                fs=self.fs,
                db=self.db,
                filename=f"{self.DATASET}_test_preprocessed_{self.model_name}",
            )
            self.generator_model = self.load_model_from_chunks(
                id_name=f"{self.DATASET}_generator_model_{self.model_name}", db=self.db
            )

        if self.MODE == "local":

            self.testA, _ = preprocess_npz_local(
                f"data/{self.DATASET}_test_preprocessed_{self.model_name}.npz"
            )
            self.generator_model = load_model(
                f"data/model/{self.DATASET}_generator_model_{self.model_name}.h5"
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

        for ix in range(self.testA.shape[0]):

            X_realA = self.testA[[ix]]
            X_fakeB = self.generator_model.predict(X_realA)
            X_fakeB = np.clip(X_fakeB * 255, 0, 255).astype(np.uint8)
            X_fakeB = X_fakeB[0]

            filename = f"{self.DATASET}_test_inference_{ix}_step_0_{self.model_name}"

            if self.MODE == "jenkins":

                image_bytes = X_fakeB.astype(np.uint8).tobytes()
                self.fs.put(image_bytes, filename=filename)

            if self.MODE == "local":

                cv2.imwrite(f"data/inference/{filename}.png", X_fakeB)


if __name__ == "__main__":

    Inference()
