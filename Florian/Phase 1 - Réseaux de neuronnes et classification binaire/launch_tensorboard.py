import os

os.system("tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 --samples_per_plugin=images=100,text=100")