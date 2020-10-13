from numpy.random import randint
from numpy import load, zeros, ones
from pathlib import Path

from keras.optimizers import Adam
from keras.models import Model, Input
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from matplotlib import pyplot
from tensorflow import keras

from train import load_real_samples, generate_fake_samples

def predict(g_model, dataset, patch_shape):
    
    trainA, trainB = dataset
    
    for ix in range(0, trainA.shape[0], 1):

        X1, X2 = trainA[[ix]], trainB[[ix]]

        X_fakeB = g_model.predict(X1)
        
        X_fakeB = (X_fakeB + 1) / 2.0
    
        pyplot.imshow(X_fakeB)

if __name__ == "__main__":

    val_dataset = load_real_samples(f"../../../data/input/model/val/val_256.npz")
    
    g_model = keras.models.load_model("../../../data/output/trained_models")
    
    predict(g_model, val_dataset, 1)
