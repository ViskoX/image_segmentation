import matplotlib as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image

class Config:
    def __init__(self):
        self.image_dir = './data/raw/COVID-19_Radiography_Dataset/COVID/images/'
        self.mask_dir = './data/raw/COVID-19_Radiography_Dataset/COVID/masks/'
        self.preprocessed_dir = './data/preprocessed/'
        self.model_dir = './checkpoints/'
        self.log_dir = './logs/'
        
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 3
        self.BATCH_SIZE = 32  
        self.EPOCHS = 8
        self.VALIDATION_SPLIT = 0.2
        self.LEARNING_RATE = 1e-4
        # mlflow params
        self.MODEL_PATH = "mlruns/0/29aec040d35c4f3783a5f34fff4dfc76/artifacts/resnet_unet_20251004-140920.h5" 
        self.IMAGE_DIR = "data/predict/"
        self.OUTPUT_DIR = "inference/"


def display_predictions(X, y_true, y_pred, num_samples=3):
    plt.figure(figsize=(15, 5*num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 4, i*4+1)
        plt.imshow(X[i])
        plt.title(f'Image {i+1}')
        plt.axis('off')
        
        # Display true mask
        plt.subplot(num_samples, 4, i*4+2)
        plt.imshow(y_true[i].squeeze(), cmap='gray')
        plt.title(f'True Mask {i+1}')
        plt.axis('off')
        
        # Display predicted mask
        plt.subplot(num_samples, 4, i*4+3)
        plt.imshow(y_pred[i].squeeze(), cmap='gray')
        plt.title(f'Pred Mask {i+1}')
        plt.axis('off')
        
        # Display image with predicted mask overlay
        plt.subplot(num_samples, 4, i*4+4)
        plt.imshow(X[i])
        plt.imshow(y_pred[i].squeeze(), alpha=0.5, cmap='Reds')
        plt.title(f'Overlay {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()




def load_image(img_path, target_size=(256, 256)):
    """
    Load an image from disk and resize it to target_size.
    Returns a NumPy array.
    """
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    return img_array

def preprocess_image(img_array):
    """
    Preprocess image for model prediction:
    - Scale pixel values to [0,1]
    - Add batch dimension
    """
    img_array = img_array / 255.0 
    return img_array  # do NOT expand dims here if batching in deploy.py



