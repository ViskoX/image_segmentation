from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation
import tensorflow as tf  
from src.utils import *

config = Config()

IMG_HEIGHT = config.IMG_HEIGHT
IMG_WIDTH = config.IMG_WIDTH
IMG_CHANNELS = config.IMG_CHANNELS




class UnetModel():
    
    def __init__(self, input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
        self.input_size = input_size
        

    def build_unet_model(self, input_size=None):

        if input_size is None:
            input_size = self.input_size
        inputs = Input(input_size)
        
        # Encoder (Contracting Path)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D((2, 2))(c4)
        
        # Bridge
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)
        
        # Decoder (Expansive Path)
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)
        
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)
        
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)
        
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)
        
        # Output layer
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    

    def build_mobilenet_unet(self, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

        # Define skip connections for MobileNetV2
        skip_names = [
            "block_1_expand_relu",   # 128x128 (if input 256x256)
            "block_3_expand_relu",   # 64x64
            "block_6_expand_relu",   # 32x32
            "block_13_expand_relu",  # 16x16
        ]
        skips = [base_model.get_layer(name).output for name in skip_names]

        encoder_output = base_model.get_layer("block_16_expand_relu").output  # 8x8

        # Decoder
        x = encoder_output
        for i in reversed(range(len(skips))):
            x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
            x = tf.keras.layers.Concatenate()([x, skips[i]])
            x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
            x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

        # 🔥 Final upsample to 256x256
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        return model