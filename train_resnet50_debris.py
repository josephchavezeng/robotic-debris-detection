# Fine-tune a pretrained ResNet-50 to classify photos as solid debris
# or no solid debris.
#
# The script reads image paths and labels from labels.csv, then divides
# the images into training and held-out groups. It trains on the training
# group and saves the model. The held-out group is evaluated separately.
#
# The source images came from HD10K's official test split, so the 80/20
# division here is a custom image-level split, not HD10K's official test.

# Path builds paths to files and folders.
from pathlib import Path

# pandas reads the labels CSV into a table and helps split the rows.
import pandas as pd

# Pillow opens image files.
from PIL import Image

# tqdm displays a progress bar during training.
from tqdm import tqdm

# PyTorch provides tensors and core machine-learning operations.
import torch

# torch.nn contains model layers and the loss function.
import torch.nn as nn

# Dataset defines how to load one example; DataLoader groups examples into batches.
from torch.utils.data import DataLoader, Dataset

# transforms resize and prepare images for ResNet-50.
from torchvision import transforms

# ResNet50_Weights provides pretrained weights; resnet50 builds the model.
from torchvision.models import ResNet50_Weights, resnet50

# train_test_split separates the data into training and held-out groups.
from sklearn.model_selection import train_test_split


# Map the text labels in labels.csv to numbers the model can use.
# 0 represents no solid debris; 1 represents solid debris.
CLASS_TO_IDX = {
    "no_solid_debris": 0,
    "solid_debris": 1,
}

# Use the same seed to make the split and training randomness repeatable.
SEED = 42

# Process 32 images in each training batch.
BATCH_SIZE = 32

# Make five complete passes through the training images.
EPOCHS = 5

# Set the size of each Adam optimizer update.
LEARNING_RATE = 1e-4

# Resize each image to 224 pixels high by 224 pixels wide.
IMAGE_SIZE = 224

# Find the folder containing this script.
PROJECT_FOLDER = Path(__file__).resolve().parent

# The prepared dataset folder is expected beside this script.
DATA_DIR = PROJECT_FOLDER / "custom_debris_dataset_420_each"

# Save the split file and model outputs beside the dataset.
OUTPUT_DIR = PROJECT_FOLDER / "outputs"


class DebrisDataset(Dataset):
    # Save the label table, image folder, and image transforms for later use.
    def __init__(self, frame, data_dir, transform=None):
        # Reset the table's row numbers so they run from 0 to the row count.
        self.frame = frame.reset_index(drop=True)

        # Keep the root folder used to locate image files.
        self.data_dir = data_dir

        # Keep the processing steps that will be applied to each image.
        self.transform = transform

    # PyTorch calls this to find out how many examples the dataset contains.
    def __len__(self):
        return len(self.frame)

    # PyTorch calls this to load one image and its label by row number.
    def __getitem__(self, index):
        # Get the table row for the requested image.
        row = self.frame.iloc[index]

        # Join the dataset folder with the relative image path from labels.csv.
        image_path = self.data_dir / row["image_path"]

        # Open the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply resizing, augmentation, and normalization transforms.
        if self.transform:
            image = self.transform(image)

        # Look up the numeric label associated with this row's class name.
        label = CLASS_TO_IDX[row["label"]]

        # Return the prepared image and a floating-point label.
        # The loss function used expects labels in this format.
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
    # Set PyTorch's random seed for repeatable model initialization
    # and batch shuffling.
    torch.manual_seed(SEED)

    # Choose the available device and show where training will run.
    device = get_device()
    print("Using device:", device)

    # Build the path to the CSV containing image paths and labels.
    labels_path = DATA_DIR / "labels.csv"

    # Read the CSV into a pandas table named frame.
    frame = pd.read_csv(labels_path)

    # Stop if the dataset contains no images.
    if frame.empty:
        raise SystemExit("No recognized class labels found in labels.csv")

    # Print the total image count and count in each class.
    print("Images:", len(frame))
    print("Class counts:")
    print(frame["label"].value_counts().to_string())

    # Divide the table into training and held-out groups.
    # test_size=0.20 reserves 20% for later evaluation.
    # random_state uses the seed so the same split is repeatable.
    # stratify keeps the two class proportions similar in both groups.
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.20,
        random_state=SEED,
        stratify=frame["label"],
    )

    # Make a copy of the full table to record which group each row belongs to.
    split_frame = frame.copy()

    # Mark all rows as training rows first.
    split_frame["split"] = "train"

    # Change the held-out rows to "test" using their original table indices.
    split_frame.loc[test_frame.index, "split"] = "test"

    # Create the output folder, including missing parent folders if needed.
    # exist_ok=True allows the folder to already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set the path for the CSV that records the train/test assignments.
    split_path = OUTPUT_DIR / "split_manifest.csv"

    # Save the assignments without writing pandas row numbers to the CSV.
    split_frame.to_csv(split_path, index=False)

    # Report the number of images assigned to each group and the saved file.
    print(f"Train images: {len(train_frame)}")
    print(f"Test images held out: {len(test_frame)}")
    print(f"Saved split manifest: {split_path}")

    # Select the standard pretrained ResNet-50 weights.
    # The weights were originally trained on ImageNet.
    weights = ResNet50_Weights.DEFAULT

    # Get the channel averages and standard deviations used with those weights.
    # The same values are used to normalize this project's images.
    mean = weights.transforms().mean
    std = weights.transforms().std

    # Build the sequence of image-processing steps used during training.
    train_transform = transforms.Compose([
        # Resize every image to the input size expected by this script.
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Randomly flip some images horizontally to vary training examples.
        transforms.RandomHorizontalFlip(),

        # Slightly vary brightness, contrast, and saturation for augmentation.
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05,
        ),

        # Convert the PIL image into a PyTorch tensor.
        transforms.ToTensor(),

        # Normalize each color channel using the pretrained model's values.
        transforms.Normalize(mean=mean, std=std),
    ])

    # Create the training dataset from training rows only.
    # The held-out test rows are not passed to the training dataset.
    train_dataset = DebrisDataset(
        train_frame,
        DATA_DIR,
        transform=train_transform,
    )

    # Create a loader that supplies batches of images to the training loop.
    train_loader = DataLoader(
        train_dataset,

        # Use the configured number of images in each batch.
        batch_size=BATCH_SIZE,

        # Shuffle the training examples so batches vary between epochs.
        shuffle=True,

        # Use the main process to load images; this is simple and works well
        # for this small dataset and avoids multiprocessing issues on some Macs.
        num_workers=0,
    )

    # Build ResNet-50 and load its pretrained ImageNet weights.
    model = resnet50(weights=weights)

    # Replace ResNet-50's original classification layer with one output value.
    # That output is a raw score logit used to distinguish the two classes.
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Move the model to the selected device.
    model = model.to(device)

    # Create the binary classification loss function.
    # It compares the model's raw score with the known 0/1 label.
    # It applies the sigmoid calculation internally.
    criterion = nn.BCEWithLogitsLoss()

    # Create the Adam optimizer, which adjusts the model's weights
    # to reduce the loss.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Create the folder where the trained checkpoint will be stored.
    model_folder = OUTPUT_DIR / "models"
    model_folder.mkdir(parents=True, exist_ok=True)

    # Set the checkpoint's filename and location.
    checkpoint_path = model_folder / "resnet50_debris.pth"

    # Repeat the training process for the configured number of epochs.
    for epoch in range(EPOCHS):

        # Tell PyTorch that the model is being trained.
        model.train()

        # Start a running total for this epoch's batch losses.
        total_loss = 0.0

        # Wrap the loader in a progress bar labeled with the current epoch.
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
        )

        # Get one batch of images and labels at a time.
        for images, labels in progress:

            # Move this batch of images to the selected device.
            images = images.to(device)

            # Move the labels to the same device and add a dimension
            # so their shape matches the model output.
            labels = labels.to(device).unsqueeze(1)

            # Clear gradients from the previous batch before calculating new ones.
            optimizer.zero_grad()

            # Run the batch through ResNet-50 to get one raw score per image.
            logits = model(images)

            # Compare the raw scores with the known labels to calculate the loss.
            loss = criterion(logits, labels)

            # Calculate gradients showing how model weights affect the loss.
            loss.backward()

            # Use the gradients to update the model weights.
            optimizer.step()

            # Add batch's average loss times its image count.
            # Weighting by batch size allows an accurate epoch average.
            total_loss += loss.item() * images.size(0)

            # Display the current batch's loss in the progress bar.
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Divide the weighted loss total by the number of training images.
        epoch_loss = total_loss / len(train_dataset)

        # Print the average training loss for this epoch.
        print(f"Train loss: {epoch_loss:.4f}")

        # Save the model weights and configuration after each epoch.
        # This preserves the latest completed model if training is interrupted.
        torch.save(
            {
                # The learned weights used to reconstruct the trained model.
                "model_state_dict": model.state_dict(),

                # The mapping from class names to numeric labels.
                "class_to_idx": CLASS_TO_IDX,

                # The input image size expected by evaluation and dashboard code.
                "image_size": IMAGE_SIZE,

                # The image normalization settings used during training.
                "normalization_mean": mean,
                "normalization_std": std,

                # The default cutoff used to turn model scores into classes.
                "threshold": 0.5,
            },
            checkpoint_path,
        )

        # Report where the checkpoint was saved.
        print(f"Saved checkpoint: {checkpoint_path}")

    # Report that all configured training epochs have completed.
    print("\nTraining complete.")

    # Clarify that the held-out group was saved but not evaluated in this script.
    print("Held-out test images were not evaluated during training.")

    # Point to the split manifest used by the separate evaluation script.
    print(f"Run evaluate_resnet50_debris.py using: {split_path}")


# Run the main function when the script is executed directly.
if __name__ == "__main__":
    main()