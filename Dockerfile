FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Définir le répertoire de travail
WORKDIR /tf

# Copier les dépendances
COPY requirements.txt /tf/requirements.txt

# Installer les outils système utiles pour le traitement d’images et Git
RUN apt-get update && apt-get install -y \
    git wget unzip ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /tf/requirements.txt

# Exposer les ports pour Jupyter et TensorBoard
EXPOSE 8888
EXPOSE 6006

# Lancement automatique de Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='root'"]
