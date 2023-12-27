from io import BytesIO

import numpy as np
import yaml
from numpy import ones, savez_compressed, zeros
from numpy.random import randint
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from googledrive import GoogleDrive


class Train:

    """
    Description: trains a GAN.
    Output: .h5 models and infered test images.
    """

    def __init__(self):

        self.config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.IMAGE_DIM = self.config["image_config"]["DIM"]
        self.LEARNING_RATE = self.config["model_config"]["LEARNING_RATE"]
        self.INPUT_FILTER = self.config["model_config"]["INPUT_FILTER"]
        self.TARGET_FILTER = self.config["model_config"]["TARGET_FILTER"]
        self.N_EPOCHS = self.config["model_config"]["N_EPOCHS"]
        self.BATCH_SIZE = self.config["model_config"]["BATCH_SIZE"]

        self.client_googledrive = GoogleDrive()

        # The main_id is the ID of the name of the dataset in Google Drive,
        # i.e. the ID of "data/bairrodorego".
        self.main_id = self.config["google_drive"]["main_id"]

        self.trained_models_id = self.client_googledrive.create_folder(
            parent_folder_id=self.main_id, folder_name="trained_models"
        )

        self.trained_models_run_id = self.client_googledrive.create_folder(
            parent_folder_id=self.trained_models_id,
            folder_name=f"{self.INPUT_FILTER}_{self.TARGET_FILTER}",
        )

        self.model_data_id = self.client_googledrive.create_folder(
            parent_folder_id=self.main_id, folder_name="model_data"
        )

        self.model_data_run_id = self.client_googledrive.create_folder(
            parent_folder_id=self.model_data_id,
            folder_name=f"{self.INPUT_FILTER}_{self.TARGET_FILTER}",
        )

        self.output_id = self.client_googledrive.create_folder(
            parent_folder_id=self.main_id, folder_name="output"
        )

        self.output_run_id = self.client_googledrive.create_folder(
            parent_folder_id=self.output_id,
            folder_name=f"{self.INPUT_FILTER}_{self.TARGET_FILTER}",
        )

        self.train_dataset = self.load_npz(dataset="train")
        self.test_dataset = self.load_npz(dataset="test")

        testA, _ = self.test_dataset

        self.images_ids = {}

        for ix in range(testA.shape[0]):

            self.images_ids[ix] = self.client_googledrive.create_folder(
                parent_folder_id=self.output_run_id, folder_name=f"image_{ix}"
            )

        self.define_discriminator()
        self.define_generator()
        self.define_gan()
        self.train_gan()

    def load_npz(self, dataset):

        item_id = self.client_googledrive.get_item_id_by_name(
            folder_id=self.model_data_run_id, file_name=f"{dataset}.npz"
        )

        data = self.client_googledrive.get_item(item_id=item_id)
        data = np.load(BytesIO(data))

        X1, X2 = data["arr_0"], data["arr_1"]

        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5

        return [X1, X2]

    def define_discriminator(self):

        init = RandomNormal(stddev=0.02)

        self.train_dataset[0].shape[1:]

        in_src_image = Input(shape=self.train_dataset[0].shape[1:])
        in_target_image = Input(shape=self.train_dataset[0].shape[1:])

        merged = Concatenate()([in_src_image, in_target_image])

        d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
            merged
        )
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
        patch_out = Activation("sigmoid")(d)

        self.discriminator_model = Model([in_src_image, in_target_image], patch_out)
        self.discriminator_model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.LEARNING_RATE, beta_1=0.5),
            loss_weights=[0.5],
        )

    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):

        init = RandomNormal(stddev=0.02)

        g = Conv2D(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)

        if batchnorm:
            g = BatchNormalization()(g, training=True)

        g = LeakyReLU(alpha=0.2)(g)

        return g

    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):

        init = RandomNormal(stddev=0.02)

        g = Conv2DTranspose(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)
        g = BatchNormalization()(g, training=True)

        if dropout:
            g = Dropout(0.5)(g, training=True)

        g = Concatenate()([g, skip_in])
        g = Activation("relu")(g)

        return g

    def define_generator(self):

        init = RandomNormal(stddev=0.02)

        in_image = Input(shape=(self.IMAGE_DIM, self.IMAGE_DIM, 3))

        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)

        b = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e7)
        b = Activation("relu")(b)

        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)

        g = Conv2DTranspose(
            3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(d7)
        out_image = Activation("tanh")(g)

        self.generator_model = Model(in_image, out_image)

    def define_gan(self):

        self.discriminator_model.trainable = False

        in_src = Input(shape=self.train_dataset[0].shape[1:])

        gen_out = self.generator_model(in_src)

        dis_out = self.discriminator_model([in_src, gen_out])

        self.gan_model = Model(in_src, [dis_out, gen_out])
        self.gan_model.compile(
            loss=["binary_crossentropy", "mae"],
            optimizer=Adam(lr=self.LEARNING_RATE, beta_1=0.5),
            loss_weights=[1, 100],
        )

    def generate_real_samples(self, dataset, n_samples, patch_shape):

        trainA, trainB = dataset
        ix = randint(0, trainA.shape[0], n_samples)

        X1, X2 = trainA[ix], trainB[ix]

        y = ones((n_samples, patch_shape, patch_shape, 1))

        return [X1, X2], y

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

            X_fakeB, y_fake = self.generate_fake_samples(
                self.generator_model, X_realA, n_patch
            )

            _ = self.discriminator_model.train_on_batch([X_realA, X_realB], y_real)

            _ = self.discriminator_model.train_on_batch([X_realA, X_fakeB], y_fake)

            _, _, _ = self.gan_model.train_on_batch(X_realA, [y_real, X_realB])

            if (i + 1) % 10 == 0:

                testA, _ = self.test_dataset

                for ix in range(testA.shape[0]):

                    X_realA = testA[[ix]]
                    X_fakeB = self.generator_model.predict(X_realA)
                    X_fakeB = (X_fakeB + 1) / 2.0
                    X_fakeB = X_fakeB.reshape(self.IMAGE_DIM, self.IMAGE_DIM, 3)

                    self.client_googledrive.export_image(
                        folder_id=self.images_ids[ix], data=X_fakeB, idx=i
                    )

        npz_data = BytesIO()
        savez_compressed(npz_data, self.discriminator_model)
        self.client_googledrive.send_bytes_file(
            folder_id=self.trained_models_run_id,
            bytes_io=npz_data,
            file_name=f"discriminator_model.h5",
        )

        npz_data = BytesIO()
        savez_compressed(npz_data, self.generator_model)
        self.client_googledrive.send_bytes_file(
            folder_id=self.trained_models_run_id,
            bytes_io=npz_data,
            file_name=f"generator_model.h5",
        )

        npz_data = BytesIO()
        savez_compressed(npz_data, self.gan_model)
        self.client_googledrive.send_bytes_file(
            folder_id=self.trained_models_run_id,
            bytes_io=npz_data,
            file_name=f"gan_model.h5",
        )


if __name__ == "__main__":

    Train()
