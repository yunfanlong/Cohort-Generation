# Semi-Supervised NLP Research @ UCSD Health

Author: Yunfan Long

## Overview
This repository is dedicated to key notebooks necessary for the training, evaluation, and deployment of the semi-supervised model.

## Notebooks Description
- **fast_training**: This notebook is a streamlined version of the `fine_training` notebook. It lacks the extensive documentation and in-depth method descriptions found in `fine_training`. Instead, it leverages methods from the `utils` directory, which contains scripts of predefined functions for model architecture.

- **interpret**: A notebook designed for comprehensive model evaluation. It encompasses a variety of evaluation metrics, provides tools for calculating saliency scores of words in test strings, and includes encoding generation utilities.

- **inference**: This notebook is responsible for applying the trained models to any provided dataset. It outputs a series of cohort files, each containing the relevant model predictions for that dataset.