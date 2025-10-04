import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_datasets
from tensorflow import keras



def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)




def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)



def bce_dice_loss(y_true, y_pred):
    # Ensure both inputs are the same type
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



def calculate_metrics(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Intersection over Union (IoU)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = intersection / (union + 1e-10)
    
    # Dice coefficient
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-10)
    
    # Pixel accuracy
    accuracy = np.mean(y_true_f == y_pred_f)
    
    # Precision and recall
    true_positives = np.sum(y_true_f * y_pred_f)
    all_positives = np.sum(y_pred_f)
    all_trues = np.sum(y_true_f)
    
    precision = true_positives / (all_positives + 1e-10)
    recall = true_positives / (all_trues + 1e-10)
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



def plot_history(history):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.plot(history.history['dice_coefficient'])
    ax1.plot(history.history['val_dice_coefficient'])
    ax1.set_title('Dice Coefficient')
    ax1.legend(['train', 'val'])
    ax1.set(xlabel='Epoch', ylabel='Dice Coefficient')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Loss')
    ax2.legend(['train', 'val'])
    ax2.set(xlabel='Epoch', ylabel='Loss')

    ax3.plot(history.history['accuracy'])
    ax3.plot(history.history['val_accuracy'])
    ax3.set_title('Accuracy')
    ax3.legend(['train', 'val'])
    ax3.set(xlabel='Epoch', ylabel='Accuracy')

    ax4.plot(history.history['dice_coefficient'])
    ax4.plot(history.history['val_dice_coefficient'])
    ax4.set_title('Dice Coefficient')
    ax4.legend(['train', 'val'])
    ax4.set(xlabel='Epoch', ylabel='Dice Coefficient')

    plt.show()    

def evaluate():
    print('Loading model...')
    model = tf.keras.models.load_model(
    './checkpoints/20251002-14130620251002-141327.h5',
    custom_objects={
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss
    },
  
)
    print('Model loaded.')
    train_ds, test_ds = get_datasets()
    print('Evaluating model...')

    history = model.evaluate(test_ds, verbose=1)
    print(history)


if __name__ == '__main__':
    print('Evaluating model...')
    evaluate()
 
