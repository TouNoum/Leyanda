import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image


def make_inference(model, class_names):
    """
    Perform inference on a sample image using the provided model and class names.
    Parameters:
    - model: Keras model for inference
    - class_names: List of class names for the dataset
    """
    print("\n--Making inference--")
    img_path = '../Dataset/Photo/photo_0001.jpg'

    print("Model input shape:", model.input_shape)

    img = image.load_img(img_path, target_size=(180, 180))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    print("Image shape before prediction:", img_array.shape)


    # Prediction
    predictions = model.predict(img_array)
    print("Raw prediction:", predictions)

    # Binary model
    predicted_class_index = int(predictions[0][0] > 0.5)
    predicted_class_name = class_names[predicted_class_index]
    print(f"Predicted class: {predicted_class_name}")


# Confusion matrix generation
def generate_confusion_matrices(test_ds):
    """
    Generate confusion matrices for all models in the specified directory.
    Parameters:
    - test_ds: TensorFlow dataset for testing
    """
    print(f"\n--Generating confusion matrices--")

    models_path = "/tf/projet/Livrable 1/models"
    weights_path = "/tf/projet/Livrable 1/weights"
    confusion_matrices_path = "/tf/projet/Livrable 1/confusion_matrices"
    os.makedirs(weights_path, exist_ok=True)

    class_names = ["Painting", "Photo", "Schematics", "Sketch", "Text"] # For multi-class
    binary_class_names = ["Non-Photo", "Photo"] # For binary classification

    for model_file in glob.glob(os.path.join(models_path, "*.keras")):
        model_name = os.path.basename(model_file).replace('.keras', '')
        print(f"Processing {model_name}")

        model = tf.keras.models.load_model(model_file)

        is_binary = model.output_shape[-1] == 1

        display_labels = binary_class_names if is_binary else class_names

        loss, accuracy = model.evaluate(test_ds)
        print(f"Test accuracy for {model_name}: {accuracy:.4f}")

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            if is_binary:  # Binary
                y_pred.extend((preds > 0.5).astype(int).flatten())
            else:  # Multiclass
                y_pred.extend(np.argmax(preds, axis=1))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        cm_path = os.path.join(confusion_matrices_path, f"{model_name}_confusion_matrix.png")
        fig.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close(fig)
