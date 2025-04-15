# Imports
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image

# Sample visualization
def visualize_class_samples(dataset, class_names, samples_per_class=5):
    """
    Visualize random samples from each class in the dataset.
    Parameters:
    - dataset: TensorFlow dataset
    - class_names: List of class names
    - samples_per_class: Number of samples to display per class
    """
    plt.figure(figsize=(15, 10))

    class_samples = {class_name: [] for class_name in class_names}

    for images, labels in dataset:
        for i, label in enumerate(labels.numpy()):
            class_name = class_names[label]
            if len(class_samples[class_name]) < samples_per_class:
                class_samples[class_name].append(images[i].numpy().astype("uint8"))

        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break

    for idx, class_name in enumerate(class_names):
        for i, sample in enumerate(class_samples[class_name]):
            plt.subplot(len(class_names), samples_per_class, idx * samples_per_class + i + 1)
            plt.imshow(sample)
            plt.axis('off')
            if i == 0:
                plt.title(class_name)

    plt.tight_layout()
    plt.show()

# Visualize class distribution
def visualize_class_distribution(dataset, class_names):
    """
    Efficiently visualize the distribution of classes in a TensorFlow dataset.
    Parameters:
    - dataset: TensorFlow dataset
    - class_names: List of class names
    """
    print("--Computing class distribution--")

    num_classes = len(class_names)
    label_counts = np.zeros(num_classes, dtype=int)

    for _, labels in dataset:
        labels_np = labels.numpy()
        label_counts += np.bincount(labels_np, minlength=num_classes)

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, label_counts)
    plt.xlabel('Classe')
    plt.ylabel('Nombre d’images')
    plt.title('Répartition des classes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()