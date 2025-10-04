from src.model import UnetModel
from src.utils import *
from src.dataset import *
from src.eval import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import datetime, os
import mlflow
import mlflow.tensorflow

config = Config()

def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=['accuracy', dice_coefficient, tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    return model

def get_callbacks(name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
        ModelCheckpoint(
            filepath=f"{name}_best.keras",
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            patience=15,
            monitor="val_dice_coefficient",
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join("logs", f"{name}_{timestamp}")),
    ]

def train_model(model, name, X_train, X_val, y_train, y_val):
    train_ds = create_dataset(X_train, y_train, config.BATCH_SIZE, is_training=True)
    val_ds = create_dataset(X_val, y_val, config.BATCH_SIZE, is_training=False)

    # Start MLflow run
    with mlflow.start_run(run_name=name):
        mlflow.tensorflow.autolog(log_models=False)  # log metrics/loss automatically, disable default model saving

        history = model.fit(
            train_ds,
            epochs=config.EPOCHS,
            validation_data=val_ds,
            callbacks=get_callbacks(name),
            verbose=1
        )

        # Save model manually as artifact
        save_path = os.path.join(
            config.model_dir,
            f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h5"
        )
        model.save(save_path)
        mlflow.log_artifact(save_path)

        mlflow.log_param("architecture", name)

        return history


if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data(config.image_dir, config.mask_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42
    )

    # Train U-Net
    unet = compile_model(UnetModel().build_unet_model())
    print("\n🚀 Training U-Net...")
    history_unet = train_model(unet, "unet", X_train, X_val, y_train, y_val) 

    # Train MobileNet-U-Net
    mobileNet = compile_model(UnetModel().build_mobilenet_unet())
    print("\n🚀 Training MobileNet-U-Net...")
    history_resnet = train_model(mobileNet, "resnet_unet", X_train, X_val, y_train, y_val)