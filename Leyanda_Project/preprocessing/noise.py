import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

# Load the /path images
def load_clear_images(raw_data_paths, img_height=180, img_width=180, test_split=0.1):
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
    print(f"\nLoading images from {raw_data_path}...")
    images = []
    for raw_data_path in raw_data_paths:
        for filename in os.listdir(raw_data_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = Image.open(os.path.join(raw_data_path, filename))
                img = img.resize((img_height, img_width))
                images.append(np.array(img))
                images_train = np.array(images[:int(len(images) * (1 - test_split))])
                images_test = np.array(images[int(len(images) * (1 - test_split)):])

    if not test_split :
        return images_train

    return images_train, images_test


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
    print(f"\nNormalizing images...")
    x_train = images_train.astype('float32') / 255.
    x_test = images_test.astype('float32') / 255.

    return x_train, x_test


def visualize_data(X, n=10):
    """
    Visualize clear data
    Parameters:
    - X: Dataset to visualize.
    - n: Number of images to visualize.
    """
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def add_noise(images, noise_factor=0.2):
    """
    Add noise to the images
    Parameters:
    - images: Dataset to add noise to.
    - noise_factor: Amount of noise to add.
    Returns:
    - noisy_images: Noisy dataset.
    """
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1., size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images


def create_denoising_model(img_height=180, img_width=180, max_pooling = False):
    """
    Create a simple model for denoising images
    Parameters:
    - img_height: Height of the images.
    - img_width: Width of the images.
    Returns:
    - autoencoder: Compiled model.
    """
    print(f"\nCreating model...")

    input_img = Input(shape=(img_height, img_width, 3)) # input image dimensions

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x) if max_pooling else x
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x) if max_pooling else x
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) if max_pooling else x
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) if max_pooling else x
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    decoded = Conv2D(3, (3, 3), activation= 'relu', padding='same')(x)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss="mae", metrics=['accuracy'])
    autoencoder.summary()

    return autoencoder


def train_denoising_model(model, x_train_noisy, x_train, x_test_noisy, x_test, batch_size=128, nb_epochs=50):
    """
    Train the denoising model
    Parameters:
    - model: Model to train.
    - x_train_noisy: Noisy training dataset.
    - x_train: Clear training dataset.
    - x_test_noisy: Noisy test dataset.
    - x_test: Clear test dataset.
    - batch_size: Batch size for training.
    - nb_epochs: Number of epochs for training.
    Returns:
    - history: Training history.
    """
    print(f"\nTraining denoising model...")
    history = model.fit(x_train_noisy, x_train,
                    epochs=nb_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    verbose=2)
    return history


def plot_denoising_training_history(history):
    """
    Plot the training history of the denoising model
    Parameters:
    - history: Training history.
    """

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()


def compare_denoised_images(x_test, decoded_imgs, x_test_noisy):
    """
    Compare original and denoised images
    Parameters:
    - x_test: Clear test dataset.
    - decoded_imgs: Denoised images.
    - x_test_noisy: Noisy test dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    axes[0].imshow(x_test[3].squeeze(), vmin=0, vmax=1)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    axes[1].imshow(decoded_imgs[3].squeeze(), vmin=0, vmax=1)
    axes[1].set_title("Decoded image")
    axes[1].axis("off")
    axes[2].imshow(x_test_noisy[3].squeeze(), vmin=0, vmax=1)
    axes[2].set_title("Noisy image")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()