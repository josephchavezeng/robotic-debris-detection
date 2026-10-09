# Evaluate the trained ResNet-50 on the held-out images from split_manifest.csv.
#
# The script loads the saved model, predicts a debris probability for each
# held-out image, compares the predictions with the labels, and calculates
# accuracy, precision, recall, F1, AUROC, and a confusion matrix.
#
# It saves the metrics to test_metrics.json and individual predictions to
# test_predictions.csv. The model is not retrained by this script.
#
# The held-out images are from a custom split of images originally selected
# from HD10K's official test split. They are not a separate official HD10K
# evaluation set.

# Path builds paths to dataset and output files.
from pathlib import Path

# json saves the evaluation metrics in a readable JSON file.
import json

# NumPy stores the collected labels and probabilities as arrays.
import numpy as np

# pandas reads the split manifest and creates the prediction table.
import pandas as pd

# Pillow opens image files.
from PIL import Image

# PyTorch runs the model and handles tensors.
import torch
import torch.nn as nn

# Dataset defines how an image and its label are loaded.
# DataLoader groups images into batches for inference.
from torch.utils.data import DataLoader, Dataset

# transforms resize and normalize images before model inference.
from torchvision import transforms

# resnet50 builds the same model architecture used during training.
from torchvision.models import resnet50

# These functions calculate classification metrics.
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Map each class name to the number used by the model.
# 0 means no solid debris; 1 means solid debris.
CLASS_TO_IDX = {
    "no_solid_debris": 0,
    "solid_debris": 1,
}

# Set the default probability cutoff for predicting solid debris.
THRESHOLD = 0.5

# Evaluate images in batches of this size.
BATCH_SIZE = 32

# Find the folder containing this script.
PROJECT_FOLDER = Path(__file__).resolve().parent

# Find the prepared images and saved training outputs beside the script.
DATA_DIR = PROJECT_FOLDER / "custom_debris_dataset_420_each"
OUTPUT_DIR = PROJECT_FOLDER / "outputs"

# Load the saved train/test assignments created by the training script.
SPLIT_PATH = OUTPUT_DIR / "split_manifest.csv"

# Load the model checkpoint created by the training script.
CHECKPOINT_PATH = OUTPUT_DIR / "models" / "resnet50_debris.pth"


class DebrisDataset(Dataset):
    # Store the table rows, image folder, and preprocessing steps.
    def __init__(self, frame, data_dir, transform=None):
        # Reset row numbers so the dataset can be indexed from 0 to len - 1.
        self.frame = frame.reset_index(drop=True)

        # Keep the root folder used to locate the image files.
        self.data_dir = data_dir

        # Keep the preprocessing steps applied to each image.
        self.transform = transform

    # Tell PyTorch how many images are in this dataset.
    def __len__(self):
        return len(self.frame)

    # Load one image and its correct label.
    def __getitem__(self, index):
        # Get the table row for the requested image.
        row = self.frame.iloc[index]

        # Combine the dataset folder and CSV image path.
        image_path = self.data_dir / row["image_path"]

        # Open the image and convert it to RGB color.
        image = Image.open(image_path).convert("RGB")

        # Apply resizing and normalization before inference.
        if self.transform:
            image = self.transform(image)

        # Convert the text class name into its numeric label, 0 or 1.
        label = CLASS_TO_IDX[row["label"]]

        # Return the prepared image and its label.
        # The label is a float because that is the format used in training.
        return image, torch.tensor(label, dtype=torch.float32)


def get_device():
    # Use an NVIDIA GPU when CUDA is available.
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Use Apple's GPU support on a Mac when MPS is available.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Use the CPU if neither GPU option is available.
    return torch.device("cpu")


def main():
    # Read the saved table that marks each image as train or test.
    split_frame = pd.read_csv(SPLIT_PATH)

    # Keep only the held-out test rows; the training rows are not evaluated.
    test_frame = split_frame[split_frame["split"] == "test"].copy()

    # Load the saved model weights and settings onto the CPU first.
    # The model will be moved to the selected device after it is rebuilt.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # Read the input image size used when the model was trained.
    image_size = int(checkpoint["image_size"])

    # Read the image normalization values saved during training.
    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]

    # Build the preprocessing steps for evaluation images.
    # Evaluation uses resizing and normalization but no random augmentation.
    transform = transforms.Compose([
        # Resize each image to the dimensions expected by the model.
        transforms.Resize((image_size, image_size)),

        # Convert the image into a PyTorch tensor.
        transforms.ToTensor(),

        # Normalize image color channels the same way as during training.
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create a dataset containing only held-out test images.
    dataset = DebrisDataset(
        test_frame,
        DATA_DIR,
        transform=transform,
    )

    # Create batches for the model to process.
    # shuffle=False preserves the order of the rows in test_frame.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Choose the available device.
    device = get_device()
    print("Using device:", device)

    # Rebuild the ResNet-50 architecture without downloading ImageNet weights.
    # The trained weights are loaded from the checkpoint below.
    model = resnet50(weights=None)

    # Replace the default output layer with the single-output layer
    # used by the debris classifier.
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Load the trained weights into the rebuilt model.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the selected device.
    model = model.to(device)

    # Put the model into evaluation mode.
    # This disables training-only behavior in layers such as dropout.
    model.eval()

    # These lists will collect one probability and one true label per image.
    all_probs = []
    all_labels = []

    # Disable gradient calculations because evaluation does not update weights.
    with torch.no_grad():

        # Process each batch of test images and their labels.
        for images, labels in loader:

            # Move the image batch to the same device as the model.
            images = images.to(device)

            # Run the images through ResNet-50 to get raw score logits.
            logits = model(images)

            # Convert the logits to probabilities between 0 and 1.
            # squeeze(1) removes the extra output dimension.
            # cpu().numpy() moves the values to NumPy for metric calculations.
            probabilities = (
                torch.sigmoid(logits)
                .squeeze(1)
                .cpu()
                .numpy()
            )

            # Add this batch's probabilities to the full results list.
            all_probs.extend(probabilities.tolist())

            # Convert this batch's true labels to ordinary integers,
            # then add them to the full labels list.
            all_labels.extend(labels.numpy().astype(int).tolist())

    # Convert all collected true labels into a NumPy integer array.
    y_true = np.asarray(all_labels, dtype=int)

    # Convert all predicted debris probabilities into a NumPy float array.
    y_prob = np.asarray(all_probs, dtype=float)

    # Turn probabilities into predicted classes using the cutoff.
    # Probability >= 0.5 becomes 1 (solid debris); below 0.5 becomes 0.
    y_pred = (y_prob >= THRESHOLD).astype(int)


    # Calculate the confusion matrix in the order of class labels [0, 1].
    # Its four values are true negatives, false positives,
    # false negatives, and true positives.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # Calculate and store the evaluation results.
    metrics = {
        # Number of held-out images evaluated.
        "test_images": int(len(y_true)),

        # Probability cutoff used to turn scores into class labels.
        "threshold": float(THRESHOLD),

        # Fraction of all predictions that were correct.
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # Of images predicted as debris, the fraction that truly had debris.
        "precision_solid_debris": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),

        # Of actual debris images, the fraction correctly predicted as debris.
        "recall_solid_debris": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),

        # Combines precision and recall into one score.
        "f1_solid_debris": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),

        # Measures how well the probabilities rank debris above no-debris cases.
        "auroc": float(roc_auc_score(y_true, y_prob)),

        # These names explain the row and column order in the matrix below.
        "confusion_matrix_labels": ["no_solid_debris", "solid_debris"],

        # Rows are actual labels; columns are predicted labels.
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }

    # Create the folder for evaluation results if it does not already exist.
    evaluation_folder = OUTPUT_DIR / "evaluation"
    evaluation_folder.mkdir(parents=True, exist_ok=True)

    # Set the output paths for the summary metrics and individual predictions.
    metrics_path = evaluation_folder / "test_metrics.json"
    predictions_path = evaluation_folder / "test_predictions.csv"

    # Open the JSON output file and write the metrics in readable format.
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Start a results table with each test image's path and original text label.
    results = test_frame[["image_path", "label"]].reset_index(drop=True).copy()

    # Add the numeric true class, debris probability, and predicted class.
    results["true_label"] = y_true
    results["debris_probability"] = y_prob
    results["predicted_label"] = np.where(
        y_pred == 1,
        "solid_debris",
        "no_solid_debris",
    )

    # Save one row per test image so predictions can be inspected later.
    results.to_csv(predictions_path, index=False)

    # Print all calculated metrics in the terminal.
    print("\nHeld-out test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Print the locations of the saved evaluation files.
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


# Run the main function when the script is executed directly.
if __name__ == "__main__":
    main()