import sys

from tensorflow.keras.models import model_from_json
from tqdm import tqdm

from encode_images import connect_to_mongodb, load_yaml
from mongodb_lib import preprocess_chunks
import cv2
from image import ImageClass

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

        trainA, trainB = preprocess_chunks(
            fs=self.fs,
            id_name=f"{self.DATASET}_test_preprocessed_{self.model_name}",
            db=self.db,
        )
        generator_model = self.load_model_from_chunks(
            id_name=f"{self.DATASET}_generator_model_{self.model_name}", db=self.db
        )

        for ix in tqdm(range(0, trainA.shape[0], 1)):

            X1, _ = trainA[[ix]], trainB[[ix]]

            X_fakeB = generator_model.predict(X1)

            X_fakeB = (X_fakeB + 1) / 2.0
            X_fakeB = X_fakeB.reshape(self.IMAGE_DIM, self.IMAGE_DIM, 3)

            X_fakeB = ImageClass(image=X_fakeB)
            image_bytes = cv2.imencode('.jpg', X_fakeB.image)[1].tobytes()
            filename = f"{self.DATASET}_test_inference_{ix}_step_0_{self.model_name}"
            self.fs.put(image_bytes, filename=filename)


if __name__ == "__main__":

    Inference()
