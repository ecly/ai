# MNIST Classifier
A very small MNIST classifier using convolutions and max pooling.
Easily trainable on CPU reaching ~98% accuracy after a few minutes of training(~30 epochs).

## Usage
```sh
poetry install # or pip install . --user
python mnist.py # optionally with --gpus 1 if available, has early stoppign
tensorboard --logdir lightning_logs # to monitor training
```
