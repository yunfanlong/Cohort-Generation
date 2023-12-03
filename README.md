# Semi-Supervised NLP Research @ UCSD Health

Author: Yunfan Long

## Overview
This repository is dedicated to key files necessary for the training, evaluation, and deployment of the semi-supervised model.

## Notebooks Description
- **fast_training.py**: This file is a streamlined version of the `fine_training.py` notebook. It lacks the extensive documentation and in-depth method descriptions found in `fine_training.py`. Instead, it leverages methods from the `utils` directory, which contains scripts of predefined functions for model architecture.

- **interpret.py**: A file designed for comprehensive model evaluation. It encompasses a variety of evaluation metrics, provides tools for calculating saliency scores of words in test strings, and includes encoding generation utilities.

- **inference.py**: This file is responsible for applying the trained models to any provided dataset. It outputs a series of cohort files, each containing the relevant model predictions for that dataset.
