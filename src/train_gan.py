import os
import sys

import cv2
import numpy as np
from numpy import ones, zeros
from numpy.random import randint
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    GaussianNoise,
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
from scipy.ndimage import laplace

class Train:

    """
    Description: trains a GAN.
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

        self.define_discriminator()
        self.define_generator()
        self.define_gan()
        self.train_gan()

    def define_discriminator(self):

        init = RandomNormal(stddev=0.02)

        self.train_dataset[0].shape[1:]

        in_src_image = Input(shape=self.train_dataset[0].shape[1:])
        in_target_image = Input(shape=self.train_dataset[0].shape[1:])

        merged = Concatenate()([in_src_image, in_target_image])
        merged = GaussianNoise(0.1)(merged)  # Add noise

        d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
            merged
        )
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
        patch_out = Activation("sigmoid")(d)

        self.discriminator_model = Model([in_src_image, in_target_image], patch_out)
        self.discriminator_model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=float(self.LEARNING_RATE), beta_1=0.5),
            loss_weights=[0.5],
        )

    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):

        init = RandomNormal(stddev=0.02)

        g = Conv2D(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)

        if batchnorm:
            g = InstanceNormalization()(g, training=True)

        g = LeakyReLU(alpha=0.2)(g)

        return g

    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):

        init = RandomNormal(stddev=0.02)

        g = Conv2DTranspose(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)
        g = InstanceNormalization()(g, training=True)

        if dropout:
            g = Dropout(0.8)(g, training=True)

        g = Concatenate()([g, skip_in])
        g = Activation("relu")(g)

        return g

    def residual_block(self, layer_input, filters):
        """Residual block with Conv2D and InstanceNormalization"""
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding="same")(layer_input)
        x = InstanceNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding="same")(x)
        x = InstanceNormalization()(x)
        return Concatenate()([x, layer_input])

    def define_generator(self):
        init = RandomNormal(stddev=0.02)

        in_image = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))

        # Add a random noise vector to the input image
        noise = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))
        noisy_in_image = Concatenate()([in_image, noise])

        e1 = self.define_encoder_block(noisy_in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)

        # Add residual blocks
        b = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e7)
        b = Activation("relu")(b)
        b = self.residual_block(b, 512)
        b = self.residual_block(b, 512)

        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512)
        d5 = self.decoder_block(d4, e3, 256)
        d6 = self.decoder_block(d5, e2, 128)
        d7 = self.decoder_block(d6, e1, 64)

        g = Conv2DTranspose(
            3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d7)
        out_image = Activation("tanh")(g)

        self.generator_model = Model([in_image, noise], out_image)

    def define_gan(self):

        # Ensure the discriminator is not trainable when training the GAN
        self.discriminator_model.trainable = False

        # Input for the source image
        in_src = Input(shape=self.train_dataset[0].shape[1:])

        # Generate random noise to concatenate with the source image
        noise = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))

        # Generate the output image using the generator
        gen_out = self.generator_model([in_src, noise])

        # Discriminator's output for the real vs generated images
        dis_out = self.discriminator_model([in_src, gen_out])

        # Feature extractor using pre-trained VGG16
        def define_feature_extractor():
            vgg = VGG16(
                include_top=False, input_shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3)
            )
            vgg.trainable = False
            return Model(inputs=vgg.inputs, outputs=vgg.layers[12].output)

        # Initialize the feature extractor
        self.feature_extractor = define_feature_extractor()

        # Extract features from both real and generated images
        gen_features = self.feature_extractor(gen_out)

        # Define the GAN model that takes in the source image and noise, and outputs:
        # - discriminator's decision,
        # - the generated image,
        # - the feature difference between real and generated images
        self.gan_model = Model([in_src, noise], [dis_out, gen_out, gen_features])

        # Compile the GAN model with the combined loss function
        self.gan_model.compile(
            loss=["binary_crossentropy", "mae", "mae"],
            optimizer=Adam(lr=float(self.LEARNING_RATE), beta_1=0.5),
            loss_weights=[0.05, 300, 1],  # Emphasize L1 loss more
        )

    def generate_real_samples(self, dataset, n_samples, patch_shape):

        trainA, trainB = dataset
        ix = randint(0, trainA.shape[0], n_samples)

        X1, X2 = trainA[ix], trainB[ix]

        y = ones((n_samples, patch_shape, patch_shape, 1))

        # Apply label smoothing
        y = y - 0.1 * np.random.random(y.shape)  # label smoothing

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

        return [X1_noisy, X2_noisy], y

    def generate_fake_samples(self, g_model, samples, patch_shape):

        X = g_model.predict(samples)
        y = zeros((len(X), patch_shape, patch_shape, 1))

        return X, y

    def train_gan(self):

        n_patch = self.discriminator_model.output_shape[1]
        trainA, _ = self.train_dataset
        bat_per_epo = int(len(trainA) / self.BATCH_SIZE)
        n_steps = bat_per_epo * self.N_EPOCHS

        for i in tqdm(list(range(n_steps))):
            [X_realA, X_realB], y_real = self.generate_real_samples(
                self.train_dataset, self.BATCH_SIZE, n_patch
            )
            noise = np.random.normal(
                size=(self.BATCH_SIZE, self.IMAGE_DIM, self.IMAGE_DIM, 3)
            )
            X_fakeB, y_fake = self.generate_fake_samples(
                self.generator_model, [X_realA, noise], n_patch
            )

            # Train discriminator
            _ = self.discriminator_model.train_on_batch([X_realA, X_realB], y_real)
            _ = self.discriminator_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # Training the GAN model
            vgg_real_features = self.feature_extractor.predict(X_realB)
            _ = self.gan_model.train_on_batch(
                [X_realA, noise], [y_real, X_realB, vgg_real_features]
            )

            # Dynamic adjustment of loss weights
            if (i + 1) % 10 == 0:  # Adjust every 1000 steps
                new_weight = max(
                    0.05, 0.1 - i / n_steps * 0.1
                )  # Adjust loss weight dynamically
                self.gan_model.compile(
                    loss=["binary_crossentropy", "mae", "mae"],
                    optimizer=Adam(lr=0.2 * float(self.LEARNING_RATE), beta_1=0.5),
                    loss_weights=[new_weight, 200, 1],  # Update the loss weights
                )

            # Save model and generate samples at specified intervals
            if (i + 1) % 1 == 0:
                testA, _ = self.test_dataset
                for ix in range(testA.shape[0]):
                    X_realA = testA[[ix]]
                    noise = np.random.normal(
                        size=(1, self.IMAGE_DIM, self.IMAGE_DIM, 3)
                    )
                    X_fakeB = self.generator_model.predict([X_realA, noise])
                    X_fakeB = np.clip(X_fakeB * 255, 0, 255).astype(np.uint8)
                    X_fakeB = X_fakeB[0]

                    # Apply Gaussian Laplace in the end for effects

                    X_fakeB = cv2.resize(
                        X_fakeB, (self.IMAGE_DIM, self.IMAGE_DIM), interpolation=cv2.INTER_LINEAR
                    )

                    X_fakeB = laplace(X_fakeB)
                    X_fakeB = cv2.bilateralFilter(X_fakeB,5,150,150)

                    filename = (
                        f"{self.DATASET}_test_evolution_{ix}_step_{i}_{self.model_name}"
                    )
                    if self.MODE == "jenkins":
                        image_bytes = X_fakeB.astype(np.uint8).tobytes()
                        self.fs.put(image_bytes, filename=filename)
                    if self.MODE == "local":
                        os.makedirs("data/evolution_gan", exist_ok=True)
                        cv2.imwrite(f"data/evolution_gan/{filename}.png", X_fakeB)

        # Save models after training
        if self.MODE == "jenkins":
            save_model(
                fs=self.fs,
                model_object=self.discriminator_model,
                model_object_name=f"{self.DATASET}_discriminator_model_{self.model_name}",
            )
            save_model(
                fs=self.fs,
                model_object=self.generator_model,
                model_object_name=f"{self.DATASET}_generator_model_{self.model_name}",
            )
            save_model(
                fs=self.fs,
                model_object=self.gan_model,
                model_object_name=f"{self.DATASET}_gan_model_{self.model_name}",
            )
        if self.MODE == "local":
            os.makedirs("data/model", exist_ok=True)
            self.discriminator_model.save(
                f"data/model/{self.DATASET}_discriminator_model_{self.model_name}.h5"
            )
            self.generator_model.save(
                f"data/model/{self.DATASET}_generator_model_{self.model_name}.h5"
            )
            self.gan_model.save(
                f"data/model/{self.DATASET}_gan_model_{self.model_name}.h5"
            )


if __name__ == "__main__":

    Train()
