import sys
sys.path.insert(0, "/tf/projet") # Add the project root directory to the Python path (docker hosting)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.inception_v3 import InceptionV3
from Leyanda_Project.preprocessing.captioning_preprocessing import preprocess_image_path


def create_image_encoder(input_shape=(299, 299, 3), embedding_dim=256):
    """
    Create an image encoder based on InceptionV3 pre-trained model.
    Parameters:
    - input_shape : Shape of the input images
    - embedding_dim : Dimension of the output embedding
    Returns:
    - encoder : Encoder model
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    output = base_model.output
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(embedding_dim, activation='relu')(output)
    encoder = Model(inputs=base_model.input, outputs=output)

    return encoder


def create_caption_decoder(vocab_size, max_length, embedding_dim, units=256):
    """
    Create a decoder model that generates captions from image features.
    Parameters:
    - vocab_size : Size of the vocabulary
    - max_length : Maximum length of captions
    - embedding_dim : Dimension of the word embeddings
    - units : Number of LSTM units
    Returns:
    - decoder : Decoder model
    """
    image_features = Input(shape=(embedding_dim,))
    caption_input = Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim,
                          mask_zero=True)(caption_input)

    h_initial = Dense(units, activation='relu', name='h_initializer')(image_features)
    c_initial = Dense(units, activation='relu', name='c_initializer')(image_features)
    lstm = LSTM(units, return_sequences=True)(embedding, initial_state=[h_initial, c_initial])

    dropout = Dropout(0.3)(lstm)
    output = Dense(vocab_size, activation='softmax')(dropout)
    decoder = Model(inputs=[image_features, caption_input], outputs=output)

    return decoder


def generate_caption(image_path, encoder_model, decoder_model, tokenizer, max_length, units=256):
    """
    Generate a caption for a given image.
    Parameters:
    - image_path : Image file path
    - encoder_model : Encoder model for feature extraction
    - decoder_model : Decoder model for caption generation
    - tokenizer : Tokenizer used to convert words to indices and vice versa
    - max_length : Maximum length of generated caption
    - units : Number of LSTM units
    Returns:
    - str : Generated caption
    """
    idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    img = preprocess_image_path(image_path)
    img = np.expand_dims(img, axis=0)

    image_features = encoder_model.predict(img, verbose=0)

    decoder_input = np.zeros((1, 1))
    decoder_input[0, 0] = tokenizer.word_index['<start>']
    decoder_h = np.zeros((1, units))
    decoder_c = np.zeros((1, units))

    generated_caption = []

    for i in range(max_length):
        predictions, decoder_h, decoder_c = decoder_model.predict(
            [decoder_input, image_features, decoder_h, decoder_c],
            verbose=0
        )
        predicted_id = np.argmax(predictions[0, 0])
        predicted_word = idx_to_word.get(predicted_id)

        if predicted_word == '<end>' or predicted_word is None:
            break

        if predicted_word not in ['<start>', '<pad>']:
            generated_caption.append(predicted_word)

        decoder_input[0, 0] = predicted_id

    return ' '.join(generated_caption)

def create_inference_model(encoder, decoder, vocab_size, units=256, embedding_dim=256):
    """
    Create a model for inference (generating captions for new images).
    Parameters:
    - encoder : Encoder model
    - decoder : Decoder model
    - vocab_size : Size of the vocabulary
    - units : Number of LSTM units
    - embedding_dim : Dimension of the word embeddings
    Returns:
    - encoder_model : Encoder model for inference
    - decoder_model : Decoder model for inference
    """
    encoder_model = encoder

    decoder_input = Input(shape=(1,), name='decoder_input')
    decoder_features_input = Input(shape=(embedding_dim,), name='decoder_features_input')
    decoder_h_state_input = Input(shape=(units,), name='decoder_h_state_input')
    decoder_c_state_input = Input(shape=(units,), name='decoder_c_state_input')

    embedding_layer = None
    lstm_layer = None
    dense_layer = None

    for layer in decoder.layers:
        if isinstance(layer, Embedding):
            embedding_layer = layer
        elif isinstance(layer, LSTM):
            lstm_layer = layer
        elif isinstance(layer, Dense) and layer.units == vocab_size:
            dense_layer = layer

    decoder_embedding = embedding_layer(decoder_input)
    decoder_outputs, h_state, c_state = lstm_layer(
        decoder_embedding,
        initial_state=[decoder_h_state_input, decoder_c_state_input]
    )
    decoder_outputs = dense_layer(decoder_outputs)

    decoder_model = Model(
        inputs=[decoder_input, decoder_features_input, decoder_h_state_input, decoder_c_state_input],
        outputs=[decoder_outputs, h_state, c_state]
    )

    return encoder_model, decoder_model


def generate_caption(image_path, encoder_model, decoder_model, tokenizer, max_length, units=256):
    """
    Generate a caption for a given image.
    Parameters:
    - image_path : Image file path
    - encoder_model : Encoder model for feature extraction
    - decoder_model : Decoder model for caption generation
    - tokenizer : Tokenizer used to convert words to indices and vice versa
    - max_length : Maximum length of generated caption
    Returns:
    - str : Generated caption
    """
    img = preprocess_image_path(image_path)
    img = np.expand_dims(img, axis=0)
    image_features = encoder_model.predict(img)
    decoder_input = np.zeros((1, 1))
    decoder_input[0, 0] = tokenizer.word_index['<start>']
    decoder_h = np.zeros((1, units))
    decoder_c = np.zeros((1, units))
    generated_caption = []

    for i in range(max_length):
        predictions, decoder_h, decoder_c = decoder_model.predict(
            [decoder_input, image_features, decoder_h, decoder_c]
        )
        predicted_id = np.argmax(predictions[0, 0])
        predicted_word = None
        idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                predicted_word = idx_to_word.get(predicted_id)
                break

        if predicted_word == '<end>' or predicted_word is None:
            break

        if predicted_word not in ['<start>', '<pad>']:
            generated_caption.append(predicted_word)

        decoder_input[0, 0] = predicted_id

    return ' '.join(generated_caption)


def loss_function(real, pred):
    """
    Custom loss function for caption generation that masks padding tokens.
    Parameters:
    - real : Actual captions
    - pred : Predicted captions
    Returns:
    - tf.Tensor : Computed loss
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy.
    Parameters:
    - history : Training history
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_example_captions(test_img_paths, test_captions, encoder, decoder, tokenizer, max_length, num_examples=5):
    """
    Display example images with their actual and predicted captions.
    Parameters:
    - test_img_paths : Paths to test images
    - test_captions : Actual captions for the test images
    - encoder : Encoder model
    - decoder : Decoder model
    - tokenizer : Tokenizer used for captions
    - max_length : Maximum length of captions
    - num_examples : Number of examples to display
    """
    encoder_model, decoder_model = create_inference_model(encoder, decoder, max_length, len(tokenizer.word_index) + 1)

    indices = np.random.choice(len(test_img_paths), num_examples, replace=False)
    plt.figure(figsize=(15, 25))

    for i, idx in enumerate(indices):
        img_path = test_img_paths[idx]
        actual_caption = test_captions[idx]
        predicted_caption = generate_caption(img_path, encoder_model, decoder_model, tokenizer, max_length)
        plt.subplot(num_examples, 1, i+1)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f'Actual: {actual_caption}\nPredicted: {predicted_caption}', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()