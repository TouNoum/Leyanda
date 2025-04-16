import os
import sys
import warnings
import logging
import tensorflow as tf

class FilteredStderr:
    """
    Filter stderr to suppress unwanted TensorFlow messages.
    """
    def __init__(self):
        self.stderr = sys.stderr
        self.blacklist = ["prefetch_autotuner", "Corrupt JPEG", "rendezvous",
                          "gpu_timer", "Out of range", "End of sequence"]

    def write(self, message):
        if not any(phrase in message for phrase in self.blacklist):
            self.stderr.write(message)

    def flush(self):
        self.stderr.flush()

def silence_tensorflow_warnings():
    """
    Suppress TensorFlow warnings and configure logging.
    """
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_DATA_EXPERIMENTAL_AUTOTUNE_BUFFERS'] = '0'

    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    sys.stderr = FilteredStderr()

    print(f"TensorFlow warnings suppression is active.")