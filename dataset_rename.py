import os
import shutil

# Chemin vers le dossier "Dataset projet"
dataset_projet_path = "./Dataset projet"

if not os.path.exists(dataset_projet_path):
    print(f"Le dossier {dataset_projet_path} n'existe pas.")
else:
    # Parcourir les sous-dossiers de "Dataset projet"
    for root, dirs, files in os.walk(dataset_projet_path):
        # Identifier les dossiers contenant "Livrable 1" dans leur nom
        for dir_name in dirs:
            if "Livrable 1" in dir_name:
                livrable_path = os.path.join(root, dir_name)
                print(f"Dossier trouvé : {livrable_path}")
                
                # Lister les sous-dossiers de ce dossier
                subdirs = [subdir for subdir in os.listdir(livrable_path) if os.path.isdir(os.path.join(livrable_path, subdir))]
                for subdir in subdirs:
                    subdir_path = os.path.join(livrable_path, subdir)
                    
                    # Déplacer chaque sous-dossier dans le dossier parent
                    new_path = os.path.join(dataset_projet_path, subdir)
                    print(f"Déplacement de '{subdir_path}' vers '{new_path}'")
                    shutil.move(subdir_path, new_path)
                
                # Vérifier si le dossier "Livrable 1" est vide après le déplacement
                if not os.listdir(livrable_path):
                    print(f"Suppression du dossier vide : {livrable_path}")
                    os.rmdir(livrable_path)