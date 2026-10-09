# Robotic Debris Detection

A Python-based computer vision pipeline integrating a fine-tuned ResNet-50 convolutional neural network (CNN) with a simulated robotic inspection system for identifying solid debris in indoor environments.

## Project Overview

This project demonstrates how computer vision and machine learning can support autonomous robotic inspection and decision-making. The pipeline consists of four stages:

1. **Dataset Preparation:** Prepare a balanced binary classification dataset using HD10K images and solid-debris annotations.
2. **Model Training:** Apply transfer learning and fine-tuning to an ImageNet-pretrained ResNet-50 model using PyTorch.
3. **Model Evaluation:** Assess classification performance on held-out images using accuracy, precision, recall, F1 score, AUROC, and a confusion matrix.
4. **Robot Simulation:** Integrate model predictions with a Streamlit dashboard that simulates robotic movement, debris detection, operator intervention, and event logging.

## Dataset Preparation

The project uses the HD10K dataset associated with *A Deep-Learning-based System for Indoor Active Cleaning* (IROS 2022).

A custom balanced dataset of **840 images** was constructed from the solid-debris bounding-box annotations:

- 420 images containing solid-debris annotations
- 420 images without solid-debris annotations

Images were randomly sampled using a fixed seed of 42. An 80/20 stratified image-level split was applied, producing 672 training images and 168 held-out test images.

The images originate from HD10K's official test split. The evaluation therefore represents a custom image-level split, not the official HD10K benchmark. Images from the same scene may share visual characteristics, so the reported metrics should not be interpreted as performance on independent environments.

The original HD10K images are not redistributed in this repository.

**Original dataset:** https://github.com/gaussianopensource/dl_active_cleaning

## Machine Learning Model

The classification model uses ResNet-50, a 50-layer residual convolutional neural network (CNN), initialized with pretrained ImageNet weights.

The original classification layer was replaced with a single-output linear layer for binary debris classification. The entire model was fine-tuned using the prepared dataset.

| Parameter | Configuration |
|---|---|
| Framework | PyTorch / torchvision |
| Architecture | ResNet-50 |
| Pretraining | ImageNet |
| Input image size | 224 × 224 |
| Training images | 672 |
| Held-out test images | 168 |
| Batch size | 32 |
| Training epochs | 5 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | BCEWithLogitsLoss |
| Decision threshold | 0.50 |

Training images were resized, randomly flipped horizontally, and augmented with small brightness, contrast, and saturation adjustments. Images were normalized using the pretrained model's ImageNet normalization parameters.

## Model Evaluation

The fine-tuned classifier was evaluated using 168 held-out images from the custom dataset split.

| Metric | Result |
|---|---|
| F1 score | **0.899** |
| AUROC | **0.958** |

Additional metrics, including accuracy, precision, recall, and the confusion matrix, are available in `outputs/evaluation/test_metrics.json`.

Individual test-image predictions and associated probabilities are saved in `outputs/evaluation/test_predictions.csv`.

## Robotic Inspection Simulation

A Streamlit dashboard integrates the trained classifier with a simulated wheeled inspection robot.

The dashboard processes saved test images as simulated camera observations and uses the predicted debris probability to determine robot behavior.

**Decision logic:**

- **Solid debris predicted:** Pause the robot and record the detection.
- **No solid debris predicted:** Advance the simulated robot along the inspection lane.
- **Operator clearance:** Resume inspection after acknowledging a debris alert.
- **Event logging:** Record timestamps, image paths, model predictions, probabilities, and robot actions.

The dashboard also visualizes robot position, inspection status, the current camera image, and classification results.

This is a software-based proof of concept. Robot motion is simulated, and no commands are transmitted to physical hardware.

## Installation

Install Python 3 and the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Running the Project

**1. Prepare the dataset**

Download and extract the original HD10K dataset so the `IROS2022_Dataset` folder is located in the project directory.

Run:

```bash
python3 build_dataset.py
```

This generates the balanced dataset and `labels.csv` manifest.

**2. Train ResNet-50**

```bash
python3 train_resnet50_debris.py
```

This creates the training/test split manifest and saves the trained model checkpoint.

**3. Evaluate the classifier**

```bash
python3 evaluate_resnet50_debris.py
```

This calculates classification metrics and saves individual predictions.

**4. Launch the robot dashboard**

```bash
python3 -m streamlit run debris_robot_dashboard.py
```

The dashboard requires the prepared image dataset, split manifest, and trained model checkpoint. These images and model weights are not included in the repository.

## Project Structure

```text
robotic-debris-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── build_dataset.py
├── train_resnet50_debris.py
├── evaluate_resnet50_debris.py
├── debris_robot_dashboard.py
└── outputs/
    ├── split_manifest.csv
    └── evaluation/
        ├── test_metrics.json
        └── test_predictions.csv
```

## Limitations and Future Work

This project demonstrates a simulated inspection pipeline rather than a deployed robotic system. The classifier was trained and evaluated on a relatively small dataset from one HD10K scene.

Potential future improvements include evaluating on independent scenes, integrating real-time camera input, implementing ROS 2 communication, adding obstacle localization, and connecting the decision logic to a physical robotic platform.

## Dataset Acknowledgment

This project uses images and annotations from HD10K, released in connection with *A Deep-Learning-based System for Indoor Active Cleaning* (IROS 2022), by the original dataset authors.

Original project: https://github.com/gaussianopensource/dl_active_cleaning
