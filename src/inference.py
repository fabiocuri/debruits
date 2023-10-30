import yaml
from tensorflow.keras.models import load_model
from tqdm import tqdm

from handlers import ImageClass
from train import load_real_samples


def predict(config, g_model, dataset, paths, mode):

    trainA, trainB = dataset

    for ix in tqdm(range(0, trainA.shape[0], 1)):

        X1, _ = trainA[[ix]], trainB[[ix]]

        X_fakeB = g_model.predict(X1)

        X_fakeB = (X_fakeB + 1) / 2.0

        imagehandler_concat = ImageClass(config=config, cv2image=X_fakeB[0], mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.get_image_name(image_name=ix)
        imagehandler_concat.export_image(output_path=f"{paths['inference']}", scale=255)


if __name__ == "__main__":

    config = yaml.load(open("./config.yaml"), Loader=yaml.FullLoader)

    folder = config["folder"]

    INPUT_FILTER = config["model_config"]["INPUT_FILTER"]
    TARGET_FILTER = config["model_config"]["TARGET_FILTER"]

    model_config = f"{INPUT_FILTER}_{TARGET_FILTER}"

    paths = {"inference": f"./data/{folder}/image/output/{model_config}/inference"}

    inference_dataset = load_real_samples(
        f"./data/{folder}/image/input/model_data/test.npz"
    )
    g_model = load_model(
        f"./data/{folder}/image/output/{model_config}/trained_models/g_model.h5"
    )

    predict(config, g_model, inference_dataset, paths, mode="test")
