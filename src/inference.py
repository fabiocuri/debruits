from tensorflow import keras
from tqdm import tqdm

from handlers import ImageClass
from train import load_real_samples


def predict(g_model, dataset, paths, mode):

    trainA, trainB = dataset

    for ix in tqdm(range(0, trainA.shape[0], 1)):

        X1, _ = trainA[[ix]], trainB[[ix]]

        X_fakeB = g_model.predict(X1)

        X_fakeB = (X_fakeB + 1) / 2.0

        imagehandler_concat = ImageClass(cv2image=X_fakeB[0], mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=ix)
        imagehandler_concat.export_image(output_path=f"{paths['inference']}", scale=255)


if __name__ == "__main__":

    paths = {"inference": "/content/drive/MyDrive/inference"}

    inference_dataset = load_real_samples("/content/debruits/data/input/model/test.npz")
    g_model = keras.models.load_model("/content/debruits/data/output/trained_models")

    predict(g_model, inference_dataset, paths, mode="test")
