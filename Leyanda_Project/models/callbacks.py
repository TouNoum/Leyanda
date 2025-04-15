# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import datetime
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter



## CALLBACK FUNCTIONS
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to log confusion matrix and table to WandB.
    """
    def __init__(self, val_data, class_names):
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for images, labels in self.val_data:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            if preds.shape[-1] == 1:  # Binary
                y_pred.extend((preds > 0.5).astype(int).flatten())
            else:  # Multiclass
                y_pred.extend(np.argmax(preds, axis=1))

        self.log_confusion_table(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(ax=ax, xticks_rotation=45)
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.tight_layout()
        wandb.log({f"conf_mat_img_epoch_{epoch + 1}": wandb.Image(fig)})
        plt.close(fig)

    def log_confusion_table(self, y_true, y_pred):
        table = wandb.Table(columns=["Actual", "Predicted", "nPredictions"])
        counts = Counter(zip(y_true, y_pred))
        for (true, pred), count in counts.items():
            table.add_data(self.class_names[true], self.class_names[pred], count)
        wandb.log({"conf_mat_table": table})

# Callback creation
def create_callbacks(model_name="default_model", tensorboard=True, early_stopping=True, model_checkpoint=True, conf_matrix=False, val_data=None, class_names=None):
    """
    Create a list of callbacks for model training.
    Parameters:
    - model_name: Name of the model
    - tensorboard: If True, adds TensorBoard callback
    - early_stopping: If True, adds EarlyStopping callback
    - model_checkpoint: If True, adds ModelCheckpoint callback
    - conf_matrix: If True, adds ConfusionMatrixCallback
    - val_data: Validation data for confusion matrix callback
    - class_names: Class names for confusion matrix callback
    Returns:
    - List of callbacks
    """
    print(f"\n--Creating callbacks--")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = []

    if tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)

    if model_checkpoint:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}{timestamp}.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(model_checkpoint_callback)

    if conf_matrix and val_data is not None and class_names is not None:
        cm_callback = ConfusionMatrixCallback(val_data=val_data, class_names=class_names)
        callbacks.append(cm_callback)

    return callbacks