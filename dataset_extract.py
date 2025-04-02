import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

# Chemin du dossier contenant les fichiers ZIP
dataset_folder = "./Dataset projet"

def extract_zip(file_name):
    zip_path = os.path.join(dataset_folder, file_name)
    extract_path = os.path.join(dataset_folder, os.path.splitext(file_name)[0])  # Créer un dossier avec le même nom que le fichier ZIP
    
    # Créer le dossier d'extraction s'il n'existe pas
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        # Extraire le fichier ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Fichier extrait : {zip_path} vers {extract_path}")
    except zipfile.BadZipFile:
        print(f"Erreur : Le fichier {zip_path} est corrompu.")

if not os.path.exists(dataset_folder):
    print(f"Le dossier {dataset_folder} n'existe pas.")
else:
    # Liste des fichiers ZIP
    zip_files = [f for f in os.listdir(dataset_folder) if f.endswith(".zip")]
    
    # Utiliser un ThreadPoolExecutor pour paralléliser l'extraction
    with ThreadPoolExecutor() as executor:
        executor.map(extract_zip, zip_files)