import yaml
from gridfs import GridFS
from tensorflow.keras.models import model_from_json
from tqdm import tqdm

from encode_images import (
    connect_to_mongodb,
    delete_all_documents_in_collection,
    load_yaml,
)
from train import preprocess_chunks


class Inference:

    """
    Description: inferes new images with the trained model.
    Output: infered images.
    """

    def __init__(self):

        self.yaml_data = load_yaml("./debruits-kubernetes/values.yaml")
        self.db = connect_to_mongodb(self.yaml_data)

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        LEARNING_RATE = self.config["model_config"]["LEARNING_RATE"]

        self.model_name = f"inputfilter_{INPUT_FILTER}_targetfilter_{TARGET_FILTER}_lr_{LEARNING_RATE}"

        self.fs = GridFS(self.db)

        self.collection_test_inference = (
            self.yaml_data[f"mongoDbtestCollectionInference"] + "_" + self.model_name
        )
        delete_all_documents_in_collection(self.db, self.collection_test_inference)
        self.collection_test_inference = self.db[self.collection_test_inference]

        self.infere()

    def load_model_from_chunks(self, id_name, db):

        file = self.fs.find_one({"filename": id_name})

        chunks_cursor = db.fs.chunks.find({"files_id": file._id}).sort("n", 1)
        model_json_bytes = b"".join(chunk["data"] for chunk in chunks_cursor)
        model_json = model_json_bytes.decode()
        model = model_from_json(model_json)

        return model

    def infere(self):

        trainA, trainB = preprocess_chunks(id_name="test.npz", db=self.db)

        generator_model = self.load_model_from_chunks(
            id_name="generator_model" + "_" + self.model_name + ".h5", db=self.db
        )

        for ix in tqdm(range(0, trainA.shape[0], 1)):

            X1, _ = trainA[[ix]], trainB[[ix]]

            X_fakeB = generator_model.predict(X1)

            X_fakeB = (X_fakeB + 1) / 2.0

            X_fakeB = X_fakeB[0]

            image_bytes = X_fakeB.tobytes()
            filename = f"image_{ix}_final"
            self.fs.put(image_bytes, filename=filename)
            image_doc = {
                "filename": filename,
                "base64_image": self.fs.get_last_version(filename)._id,
            }
            self.collection_test_inference.insert_one(image_doc)


if __name__ == "__main__":

    Inference()
