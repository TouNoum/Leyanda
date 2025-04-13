import os
import tensorflow as tf

# Dossier contenant les fichiers .keras
KERAS_MODELS_DIR = "./models"
# Dossier de sortie des poids .h5
WEIGHTS_DIR = "weights"

# Création du dossier de sortie s’il n’existe pas
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Liste des fichiers .keras dans le dossier
keras_files = [f for f in os.listdir(KERAS_MODELS_DIR) if f.endswith(".keras")]

for keras_file in keras_files:
    keras_path = os.path.join(KERAS_MODELS_DIR, keras_file)
    
    print(f"Chargement du modèle : {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    # Nom du fichier de sortie
    base_name = os.path.splitext(keras_file)[0]
    weights_path = os.path.join(WEIGHTS_DIR, f"{base_name}.weights.h5")
    
    print(f"Sauvegarde des poids dans : {weights_path}")
    model.save_weights(weights_path)

print("✅ Sauvegarde des poids terminée.")
