# Imports
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
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Lambda, Concatenate, Add, Multiply
from tensorflow.keras.applications.inception_v3 import InceptionV3
from Leyanda_Project.preprocessing.captioning_preprocessing import preprocess_image_path


def create_image_encoder(input_shape=(180, 180, 3), embedding_dim=512, fine_tune_layers=30):
    """
    Create an image encoder based on InceptionV3 pre-trained model with fine-tuning capability.
    Parameters:
    - input_shape : Shape of the input images
    - embedding_dim : Dimension of the output embedding
    - fine_tune_layers : Number of layers to fine-tune from the end
    Returns:
    - encoder : Encoder model
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    if fine_tune_layers > 0:
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True

    output = base_model.output
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(embedding_dim, activation='relu')(output)
    encoder = Model(inputs=base_model.input, outputs=output)

    return encoder


def create_caption_decoder_basic(vocab_size, max_length, embedding_dim, units=256, dropout_rate=0.3):
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

    dropout = Dropout(dropout_rate)(lstm)
    output = Dense(vocab_size, activation='softmax')(dropout)
    decoder = Model(inputs=[image_features, caption_input], outputs=output)

    return decoder


def create_caption_decoder_with_attention(vocab_size, max_length, embedding_dim, units=256, dropout_rate=0.5):
    """
    Create a decoder model with a simplified attention mechanism.
    """
    image_features = Input(shape=(embedding_dim,))
    caption_input = Input(shape=(max_length,))

    image_features_expanded = Lambda(lambda x: tf.expand_dims(x, 1))(image_features)
    image_features_repeated = Lambda(lambda x: tf.repeat(x, repeats=max_length, axis=1))(image_features_expanded)

    embedding = Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim,
                          mask_zero=True)(caption_input)

    lstm_out = LSTM(units, return_sequences=True, dropout=dropout_rate)(embedding)

    combined = Concatenate(axis=-1)([lstm_out, image_features_repeated])
    attention = Dense(units, activation='tanh')(combined)
    attention = Dense(1, activation='softmax')(attention)
    weighted_sum = Multiply()([lstm_out, attention])

    dropout_out = Dropout(dropout_rate)(weighted_sum)
    output = Dense(vocab_size, activation='softmax')(dropout_out)

    decoder = Model(inputs=[image_features, caption_input], outputs=output)
    return decoder

def create_captioning_model(encoder, decoder, max_length):
    """
    Create the complete image captioning model by connecting encoder and decoder.
    Parameters:
    - encoder : Encoder model
    - decoder : Decoder model
    - max_length : Maximum length of captions
    Returns:
    - captioning_model : Complete model for image captioning
    """
    image_input = Input(shape=(180, 180, 3), name='image_input')
    caption_input = Input(shape=(max_length,), name='caption_input')
    image_features = encoder(image_input)
    caption_output = decoder([image_features, caption_input])
    captioning_model = Model(
        inputs=[image_input, caption_input],
        outputs=caption_output,
        name='captioning_model'
    )

    return captioning_model

def generate_caption_basic(image_path, model, tokenizer, max_length=30):
    """
    Generate caption for an image using the trained model directly.
    Parameters:
    - image_path : Path to the image
    - model : Trained model
    - tokenizer : Tokenizer used for captions
    - max_length : Maximum length of captions
    Returns:
    - caption : Generated caption for the image
    """
    img = preprocess_image_path(image_path)
    img = np.expand_dims(img, axis=0)
    caption = ['<start>']

    for i in range(max_length - 1):
        sequence = tokenizer.texts_to_sequences([' '.join(caption)])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        sequence = np.expand_dims(sequence, axis=0)

        pred = model.predict([img, sequence], verbose=0)
        pred_idx = np.argmax(pred[0, i, :])

        if pred_idx == tokenizer.word_index.get('<end>', 1) or pred_idx == 0:
            break

        for word, idx in tokenizer.word_index.items():
            if idx == pred_idx:
                caption.append(word)
                break

    caption = caption[1:]

    if '<end>' in caption:
        caption = caption[:caption.index('<end>')]

    return ' '.join(caption)

def generate_caption_diverse(image_path, model, tokenizer, max_length=30, temperature=0.4):
    """
    Generate caption with temperature parameter to increase diversity.
    Parameters:
    - image_path : Path to the image
    - model : Trained model
    - tokenizer : Tokenizer used for captions
    - max_length : Maximum length of captions
    - temperature : Temperature parameter for sampling, by default 0.7
    Returns:
    - caption : Generated caption for the image
    """
    img = preprocess_image_path(image_path)
    img = np.expand_dims(img, axis=0)
    caption = ['<start>']

    max_iterations = max_length * 2
    iteration = 0

    for i in range(max_length - 1):
        if iteration >= max_iterations:
            break

        iteration += 1
        sequence = tokenizer.texts_to_sequences([' '.join(caption)])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        sequence = np.expand_dims(sequence, axis=0)

        pred = model.predict([img, sequence], verbose=0)

        if temperature > 0:
            logits = np.log(pred[0, min(i, pred.shape[1]-1), :]) / temperature
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probs = exp_logits / np.sum(exp_logits)
            pred_idx = np.random.choice(len(probs), p=probs)
        else:
            pred_idx = np.argmax(pred[0, min(i, pred.shape[1]-1), :])

        if pred_idx == tokenizer.word_index.get('<end>', 1) or pred_idx == 0:
            break

        word_found = False
        for word, idx in tokenizer.word_index.items():
            if idx == pred_idx:
                caption.append(word)
                word_found = True
                break

        if not word_found:
            break

    caption = [word for word in caption if word not in ['<start>', '<end>']]

    return ' '.join(caption)


def basic_loss(real, pred):
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


def regularized_semantic_loss(real, pred):
    """
    Semantic loss function with regularization to combat overfitting.
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

    semantic_boost = 0.05 * tf.reduce_mean(tf.math.reduce_std(pred, axis=-1))

    l2_loss = 0.001 * tf.reduce_sum(tf.square(pred))

    return tf.reduce_mean(loss_) - semantic_boost + l2_loss


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