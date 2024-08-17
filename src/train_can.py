import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from numpy.random import randint
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.layers import InstanceNormalization
from tqdm import tqdm

from mongodb_lib import (
    connect_to_mongodb,
    load_yaml,
    preprocess_npz,
    preprocess_npz_local,
    save_model,
)


class Train:

    """
    Description: trains a Creativity-Augmented Network (CAN).
    Output: models and infered test images.
    """

    def __init__(self):

        self.config = load_yaml(yaml_path="config_pipeline.yaml")

        self.MODE = sys.argv[1]
        self.DATASET = sys.argv[2]
        self.INPUT_FILTER = sys.argv[3]
        self.TARGET_FILTER = sys.argv[4]
        self.LEARNING_RATE = sys.argv[5]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]
        self.N_EPOCHS = self.config["model_config"]["N_EPOCHS"]
        self.BATCH_SIZE = self.config["model_config"]["BATCH_SIZE"]

        self.model_name = (
            f"{self.INPUT_FILTER}_{self.TARGET_FILTER}_{self.LEARNING_RATE}"
        )

        if self.MODE == "jenkins":

            self.db, self.fs = connect_to_mongodb(config=self.config)
            self.train_dataset = preprocess_npz(
                fs=self.fs,
                db=self.db,
                filename=f"{self.DATASET}_train_preprocessed_{self.model_name}",
            )
            self.test_dataset = preprocess_npz(
                fs=self.fs,
                db=self.db,
                filename=f"{self.DATASET}_test_preprocessed_{self.model_name}",
            )

        if self.MODE == "local":

            self.train_dataset = preprocess_npz_local(
                f"data/{self.DATASET}_train_preprocessed_{self.model_name}.npz"
            )
            self.test_dataset = preprocess_npz_local(
                f"data/{self.DATASET}_test_preprocessed_{self.model_name}.npz"
            )

        self.define_encoder()
        self.define_decoder()
        self.define_feature_extractor()
        self.define_can()  # Ensure this method is called
        self.train_can()

    def define_encoder(self):
        init = RandomNormal(mean=0.0, stddev=0.02, seed=42)  # Add a seed here

        in_image = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))

        e1 = Conv2D(
            64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(in_image)
        e1 = LeakyReLU(alpha=0.2)(e1)
        e1 = InstanceNormalization()(e1)

        e2 = Conv2D(
            128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e1)
        e2 = LeakyReLU(alpha=0.2)(e2)
        e2 = InstanceNormalization()(e2)

        e3 = Conv2D(
            256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e2)
        e3 = LeakyReLU(alpha=0.2)(e3)
        e3 = InstanceNormalization()(e3)

        e4 = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e3)
        e4 = LeakyReLU(alpha=0.2)(e4)
        e4 = InstanceNormalization()(e4)

        self.encoder_model = Model(in_image, e4)

    def define_decoder(self):
        init = RandomNormal(mean=0.0, stddev=0.02, seed=42)  # Add a seed here

        in_encoded = Input(shape=(self.IMAGE_DIM // 16, self.IMAGE_DIM // 16, 512))

        d1 = Conv2DTranspose(
            256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(in_encoded)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = InstanceNormalization()(d1)

        d2 = Conv2DTranspose(
            128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = InstanceNormalization()(d2)

        d3 = Conv2DTranspose(
            64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = InstanceNormalization()(d3)

        d4 = Conv2DTranspose(
            3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d3)
        out_image = Activation("tanh")(d4)

        self.decoder_model = Model(in_encoded, out_image)

    def define_feature_extractor(self):
        """Define a feature extractor using VGG16 for style and content loss."""
        vgg = VGG16(include_top=False, input_shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))
        vgg.trainable = False
        outputs = [
            vgg.get_layer(name).output
            for name in ["block2_conv2", "block3_conv3", "block4_conv3"]
        ]
        self.feature_extractor = Model(inputs=vgg.inputs, outputs=outputs)

    def compute_feature_loss(self, real_features, generated_features):
        """Compute the feature loss based on extracted features."""
        loss = 0
        for real_feat, gen_feat in zip(real_features, generated_features):
            loss += np.mean(np.abs(real_feat - gen_feat))
        return loss

    def define_can(self):
        """Define the Creativity-Augmented Network (CAN) model."""
        in_image = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))
        encoded = self.encoder_model(in_image)
        decoded = self.decoder_model(encoded)

        self.can_model = Model(in_image, decoded)
        self.can_model.compile(
            loss=self.feature_loss,
            optimizer=Adam(learning_rate=float(self.LEARNING_RATE)),
        )

    def feature_loss(self, y_true, y_pred):
        # Use model call directly instead of `predict`
        real_features = self.feature_extractor(
            y_true, training=False
        )  # Make sure `training=False` if using BatchNormalization or Dropout
        fake_features = self.feature_extractor(y_pred, training=False)

        # Compute the feature loss
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            # Ensure that real_feat and fake_feat are tensors
            real_feat = tf.convert_to_tensor(real_feat)
            fake_feat = tf.convert_to_tensor(fake_feat)

            # Compute the feature loss for each feature map
            loss += tf.reduce_mean(tf.abs(real_feat - fake_feat))

        return loss

    def generate_real_samples(self, dataset, n_samples):

        trainA, trainB = dataset
        ix = randint(0, trainA.shape[0], n_samples)

        X1, X2 = trainA[ix], trainB[ix]

        # Add noise to the real images
        noise = np.random.normal(scale=0.05, size=X1.shape)
        X1_noisy = X1 + noise
        X2_noisy = X2 + noise

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        # In generate_real_samples, augment data before returning
        X1_noisy = next(
            datagen.flow(X1_noisy, batch_size=self.BATCH_SIZE, shuffle=False)
        )
        X2_noisy = next(
            datagen.flow(X2_noisy, batch_size=self.BATCH_SIZE, shuffle=False)
        )

        return X1_noisy, X2_noisy

    def train_can(self):

        trainA, _ = self.train_dataset
        bat_per_epo = int(len(trainA) / self.BATCH_SIZE)
        n_steps = bat_per_epo * self.N_EPOCHS

        for i in tqdm(range(n_steps)):
            X_realA, X_realB = self.generate_real_samples(
                self.train_dataset, self.BATCH_SIZE
            )

            # Train the CAN model
            _ = self.can_model.train_on_batch(X_realA, X_realB)

            # Save model and generate samples at specified intervals
            if (i + 1) % 1 == 0:
                testA, _ = self.test_dataset
                for ix in range(testA.shape[0]):
                    X_realA = testA[[ix]]
                    X_fakeB = self.can_model.predict(X_realA)
                    X_fakeB = np.clip(X_fakeB * 255, 0, 255).astype(np.uint8)
                    X_fakeB = X_fakeB[0]

                    filename = (
                        f"{self.DATASET}_test_evolution_{ix}_step_{i}_{self.model_name}"
                    )
                    if self.MODE == "jenkins":
                        image_bytes = X_fakeB.astype(np.uint8).tobytes()
                        self.fs.put(image_bytes, filename=filename)
                    if self.MODE == "local":
                        os.makedirs("data/evolution_can", exist_ok=True)
                        cv2.imwrite(f"data/evolution_can/{filename}.png", X_fakeB)

        # Save models after training
        if self.MODE == "jenkins":
            save_model(
                fs=self.fs,
                model_object=self.can_model,
                model_object_name=f"{self.DATASET}_can_model_{self.model_name}",
            )
        if self.MODE == "local":
            os.makedirs("data/model", exist_ok=True)
            self.can_model.save(
                f"data/model/{self.DATASET}_can_model_{self.model_name}.h5"
            )


if __name__ == "__main__":

    Train()
