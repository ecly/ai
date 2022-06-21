# Deep Q Learning Course

My code from following the course "Modern Reinforcement Learning: Deep Q Learning in PyTorch" on Udemy:
https://www.udemy.com/course/deep-q-learning-from-paper-to-code

Not the prettiest code or most optimized code. Only focused on quickly demonstrating functionality for all exercises.

## Usage
```sh
poetry install # or pip install . --user

# on my machine with cuda 11+ in WSL, I install torch with
poetry run pip install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```
An exercise can be run using `poetry run python -m <section>.exercise<number>` e.g. `poetry run python -m section2.exercise3`
