# Leyanda

Build image docker:
```
docker build -t tensorflow-jupyter:latest .
```
Run conteneur docker:
```
docker run -p 8888:8888 -p 6006:6006 -v $(pwd):/tf/work tensorflow-jupyter:latest
```