from numpy.random import randint
from numpy import load, zeros, ones
from pathlib import Path
from pathlib import Path
from keras.optimizers import Adam
from keras.models import Model, Input
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from matplotlib import pyplot
from tensorflow import keras
import cv2

from handlers import ImageClass

from train import load_real_samples, generate_fake_samples

def predict(g_model, dataset, paths, mode):
    
    trainA, trainB = dataset
    
    for ix in range(0, trainA.shape[0], 1):

        X1, X2 = trainA[[ix]], trainB[[ix]]

        X_fakeB = g_model.predict(X1)
        
        X_fakeB = (X_fakeB + 1) / 2.0
        
        imagehandler_concat = ImageClass(cv2image=X_fakeB[0], mode=mode)
        imagehandler_concat.read_image()
        imagehandler_concat.imshow()
        cv2.waitKey(0)
        imagehandler_concat.get_image_name(image_name=ix)
        imagehandler_concat.export_image(output_path=f"{paths['inference']}")

if __name__ == "__main__":

    input_path = "../../../data/input"

    paths = {"inference": f"{input_path}/inference"}

    inference_dataset = load_real_samples(f"../../../data/input/model/test.npz")
    g_model = keras.models.load_model("../../../data/output/trained_models")
    
    predict(g_model, inference_dataset, paths, mode='test')
