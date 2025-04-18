import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# Load the /path images
def load_clear_images(raw_data_path, img_height=180, img_width=180, test_split=0.1):
    """
    Assemble a dataset from the Dataset folder
    - raw_data_path: Path to the raw data folder.
    - img_height: Height of the images.
    - img_width: Width of the images.
    - test_split: Proportion of the dataset used for testing.
    Returns:
    - images_train: Training dataset.
    - images_test: Test dataset.
    """
    print(f"\n--Loading images from {raw_data_path}--")
    images = []
    for filename in os.listdir(raw_data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(raw_data_path, filename))
            img = img.resize((img_height, img_width))
            images.append(np.array(img))
            images_train = np.array(images[:int(len(images) * (1 - test_split))])
            images_test = np.array(images[int(len(images) * (1 - test_split)):])

    return images_train, images_test

# Image normalization
def normalize_image(images_train, images_test):
    """
    Normalize the images using mean and standard deviation
    Parameters:
    - images_train: Training dataset.
    - images_test: Test dataset.
    Returns:
    - x_train: Normalized training dataset.
    - x_test: Normalized test dataset.
    """
    print(f"\n--Normalizing images--")
    x_train = images_train.astype('float32') / 255.
    x_test = images_test.astype('float32') / 255.

    return x_train, x_test

# Visualization
def visualize_data(X, n=10):
    """
    Visualize clear data
    Parameters:
    - X: Dataset to visualize.
    - n: Number of images to visualize.
    """
    print(f"\n--Visualizing data--")
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()