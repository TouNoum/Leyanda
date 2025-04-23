import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.inception_v3 import InceptionV3


def load_coco_dataset(images_folder, annotations_folder, annotation_file="captions_train2017.json"):
    """
    Load the COCO dataset with images and their captions.
    Parameters:
    - images_folder : Path to the folder containing images
    - annotations_folder : Path to the folder containing annotations
    - annotation_file : Name of the annotation file, by default "captions_train2017.json"
    Returns:
    - image_paths : List of image paths
    - captions : List of corresponding captions
    """
    print(f"Loading COCO dataset from {images_folder} and {annotations_folder}...")

    annotations_path = os.path.join(annotations_folder, annotation_file)
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)

    image_paths = []
    captions = []

    for annotation in annotations_data['annotations']:
        img_id = annotation['image_id']
        img_name = f'{int(img_id):012d}.jpg'
        img_path = os.path.join(images_folder, img_name)

        if os.path.exists(img_path):
            image_paths.append(img_path)
            captions.append(annotation['caption'])

    print(f"Loaded {len(image_paths)} images with captions")
    return image_paths, captions


def create_tokenizer(captions, num_words=10000):
    """
    Create and fit a tokenizer on all captions.
    Parameters:
    - captions : List of captions
    - num_words : Maximum number of words to keep, by default 10000
    Returns:
    - tokenizer : Fitted tokenizer
    - vocab_size : Size of the vocabulary
    """
    print("Creating and fitting tokenizer...")

    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
    )

    processed_captions = ['<start> ' + caption + ' <end>' for caption in captions]

    tokenizer.fit_on_texts(processed_captions)

    word_index = tokenizer.word_index
    if '<start>' not in word_index:
        word_index['<start>'] = len(word_index) + 1
    if '<end>' not in word_index:
        word_index['<end>'] = len(word_index) + 1

    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")

    return tokenizer, vocab_size


def preprocess_caption(caption, tokenizer, max_length=30):
    """
    Preprocess a caption: add tokens, convert to sequence and pad.
    Parameters:
    - caption : Caption to preprocess
    - tokenizer : Fitted tokenizer
    - max_length : Maximum caption length, by default 30
    Returns:
    - padded_sequence : Padded sequence of the caption
    """
    caption = '<start> ' + caption + ' <end>'

    sequence = tokenizer.texts_to_sequences([caption])[0]
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]

    return padded_sequence


def create_dataset_generator(image_paths, captions, tokenizer, max_length=30, batch_size=32, shuffle=True):
    """
    Create a TensorFlow data generator that yields batches of preprocessed images and captions.
    Parameters:
    - image_paths : List of image paths
    - captions : List of corresponding captions
    - tokenizer : Fitted tokenizer
    - max_length : Maximum caption length, by default 30
    - batch_size : Size of the batches, by default 32
    - shuffle : Whether to shuffle the dataset, by default True
    Returns:
    - dataset : A TensorFlow dataset object
    """
    def generator():
        indices = list(range(len(image_paths)))
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            img_path = image_paths[i]
            caption = captions[i]
            img = preprocess_image_path(img_path, target_size=(299, 299))
            cap_input = preprocess_caption(caption, tokenizer, max_length)
            cap_target = cap_input[1:]  # Supprime le token <start>
            cap_target = np.append(cap_target, 0)

            yield (img, cap_input), cap_target

    output_signature = (
        (
            tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def split_dataset(image_paths, captions, train_split=0.8, val_split=0.1):
    """
    Split the dataset into training, validation and test sets.
    - image_paths : List of image paths
    - captions : List of corresponding captions
    - train_split : Proportion of the dataset to use for training, by default 0.8
    - val_split : Proportion of the dataset to use for validation, by default 0.1
    Returns:
    - train_img_paths : List of training image paths
    - train_captions : List of training captions
    - val_img_paths : List of validation image paths
    - val_captions : List of validation captions
    - test_img_paths : List of test image paths
    - test_captions : List of test captions
    """
    print("Splitting dataset into train, validation, and test sets...")

    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    train_size = int(train_split * len(image_paths))
    val_size = int(val_split * len(image_paths))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_img_paths = [image_paths[i] for i in train_indices]
    train_captions = [captions[i] for i in train_indices]

    val_img_paths = [image_paths[i] for i in val_indices]
    val_captions = [captions[i] for i in val_indices]

    test_img_paths = [image_paths[i] for i in test_indices]
    test_captions = [captions[i] for i in test_indices]

    print(f"Train set: {len(train_img_paths)} samples")
    print(f"Validation set: {len(val_img_paths)} samples")
    print(f"Test set: {len(test_img_paths)} samples")

    return (train_img_paths, train_captions,
            val_img_paths, val_captions,
            test_img_paths, test_captions)