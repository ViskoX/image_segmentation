import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from src.utils import Config
import os
from sklearn.model_selection import train_test_split


config = Config()

IMG_HEIGHT = config.IMG_HEIGHT
IMG_WIDTH = config.IMG_WIDTH
IMG_CHANNELS = config.IMG_CHANNELS
BATCH_SIZE = config.BATCH_SIZE
VALIDATION_SPLIT = config.VALIDATION_SPLIT

def load_data(image_path, mask_path):
    image_files = sorted(glob(os.path.join(image_path, '*.*')))
    mask_files = sorted(glob(os.path.join(mask_path, '*.*')))
    
    assert len(image_files) == len(mask_files), "Number of images and masks doesn't match!"
    print(f"Found {len(image_files)} images and masks")
    
    # Initialize arrays
    X = np.zeros((len(image_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    y = np.zeros((len(mask_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    
    # Load images and masks
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        if i % 1000 == 0:
            print(f"Processing image {i}/{len(image_files)}")
            
        # Load and resize image
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X[i] = img / 255.0  
        
        # Load and resize mask
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        # Convert grayscale to binary (assuming mask is already binary)
        mask = (mask > 128).astype(np.float32)
        y[i] = np.expand_dims(mask, axis=-1)
    
    return X, y


def create_dataset(X, y, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        # Shuffle the dataset for training
        dataset = dataset.shuffle(buffer_size=len(X))
        
        # Data augmentation using tf.data - more efficient on GPU
        def augment(image, mask):
            # Random flip left-right
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)
            
            # Random flip up-down
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)
            
            # Random brightness adjustment
            image = tf.image.random_brightness(image, 0.1)
            
            # Ensure image values stay in [0,1]
            image = tf.clip_by_value(image, 0, 1)
            
            return image, mask
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def save_preprocessed_data_numpy():
    save_dir = config.preprocessed_dir
    
    def preprocessed_files_exist(folder_path):
        """Check if all expected preprocessed files exist"""
        required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
        return all(os.path.exists(os.path.join(folder_path, f)) for f in required_files)
    
    # Check if preprocessed files already exist
    if os.path.exists(save_dir) and preprocessed_files_exist(save_dir):
        print(f"Preprocessed data already exists in {save_dir}. Skipping processing.")
        return
    
    """Save preprocessed data as numpy arrays"""
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = load_data(config.image_dir, config.mask_dir)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    
    print(f"Data saved to {save_dir}/")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

def load_preprocessed_data_numpy():
    save_dir=config.preprocessed_dir
    """Load preprocessed numpy arrays"""
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(save_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(save_dir, 'y_val.npy'))
    
    print(f"Loaded - Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def get_datasets():
    print('Loading preprocessed data...')
    save_preprocessed_data_numpy()
    X_train, X_val, y_train, y_val = load_preprocessed_data_numpy()
    train_ds = create_dataset(X_train, y_train, BATCH_SIZE, is_training=True)
    val_ds = create_dataset(X_val, y_val, BATCH_SIZE, is_training=False)

    return train_ds, val_ds

