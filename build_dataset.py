# Build a balanced solid-debris classification dataset from HD10K.
#
# The script looks in test/scene_1 for annotations and matching photos.
# It selects 420 photos with solid-debris boxes and 420 with blank
# solid-debris annotation files, then copies them into class folders.
# It also writes a labels.csv manifest with each photo's label and source.
#
# The selected photos come from HD10K's official test split. 
# Later evaluation uses a custom split of these photos, not HD10K's official test.

# Path handles file and folder locations.
from pathlib import Path

# csv writes the labels manifest in CSV format.
import csv

# random selects a repeatable sample from each class.
import random

# shutil copies the selected photo files.
import shutil


# Find the folder containing this script, then build paths relative to it.
PROJECT_FOLDER = Path(__file__).resolve().parent

# Location of the extracted HD10K dataset.
DATASET_FOLDER = PROJECT_FOLDER / "IROS2022_Dataset"

# Location where new class folders and labels.csv will be created.
OUTPUT_FOLDER = PROJECT_FOLDER / "custom_debris_dataset_420_each"

# Use annotations and photos from this scene.
SCENE = "scene_1"

# Select this many images for each class.
IMAGES_PER_CLASS = 420

# Use the same seed to make random selection repeatable across runs.
RANDOM_SEED = 42


def has_box_rows(annotation_file):
    # Read the annotation file as text and split it into separate lines.
    for line in annotation_file.read_text().splitlines():

        # Remove whitespace from both ends of the current line.
        line = line.strip()

        # A nonempty line that does not start with # counts as an annotation row.
        if line and not line.startswith("#"):

            # Stop checking and report that this file contains a row.
            return True

    # No annotation rows were found, so report that the file is blank.
    return False


def main():
    # Build the path to HD10K's test solid-debris annotation folder.
    boxes_folder = DATASET_FOLDER / "test" / "solid_dirts_bboxes"

    # Build the path to the corresponding test photos folder.
    images_folder = DATASET_FOLDER / "test" / "images"

    # These lists will hold (annotation path, photo path) pairs.
    # Each pair will be sorted into one of the two classes below.
    debris_images = []
    no_debris_images = []

    # Add the selected scene name to the annotation folder path.
    scene_boxes = boxes_folder / SCENE

    # Find every .txt annotation in the selected scene.
    # sorted() to make the file processing order consistent.
    for annotation_file in sorted(scene_boxes.rglob("*.txt")):

        # Skip macOS metadata files; they are not dataset annotations.
        if annotation_file.name.startswith("._"):
            continue

        # Find the annotation's path relative to the annotation folder.
        relative_path = annotation_file.relative_to(boxes_folder)

        # Build the matching photo path by changing .txt to .jpg.
        image_file = images_folder / relative_path.with_suffix(".jpg")

        # If the matching photo does not exist, skip this annotation.
        if not image_file.is_file():
            continue

        # Keep this annotation path and its matching photo path together.
        pair = (annotation_file, image_file)

        # Check whether the annotation contains any box rows.
        if has_box_rows(annotation_file):

            # Store pairs whose annotation contains one or more box rows.
            debris_images.append(pair)
        else:

            # Store pairs whose annotation contains no solid-debris box rows.
            no_debris_images.append(pair)

    # Verify enough images are available for each class.
    if len(debris_images) < IMAGES_PER_CLASS:
        print("Not enough images with debris boxes.")
        return

    if len(no_debris_images) < IMAGES_PER_CLASS:
        print("Not enough images with blank box files.")
        return

    # Set the seed so the sample is repeatable.
    random.seed(RANDOM_SEED)

    # Randomly select the requested number of debris examples.
    # Each selected pair is stored with the class label "solid_debris".
    selected = [
        ("solid_debris", pair)
        for pair in random.sample(debris_images, IMAGES_PER_CLASS)
    ]

    # Randomly select the same number of blank-annotation examples.
    # Add each pair to the existing selected list with its class label.
    selected += [
        ("no_solid_debris", pair)
        for pair in random.sample(no_debris_images, IMAGES_PER_CLASS)
    ]

    # This list will hold the rows that will be written to labels.csv.
    csv_rows = []

    # Process each selected item.
    # Each item contains a label and its annotation/photo path pair.
    for label, (annotation_file, image_file) in selected:

        # Build the destination class folder
        class_folder = OUTPUT_FOLDER / label

        # Create the class folder and any missing parent folders.
        # exist_ok=True allows the folder to already exist.
        # Existing files are not removed, so use an empty output folder.
        class_folder.mkdir(parents=True, exist_ok=True)

        # Keep the photo's original filename inside its class folder.
        destination = class_folder / image_file.name

        # Copy the source photo to the new dataset folder.
        shutil.copy(image_file, destination)

        # Add one manifest record:
        # the new photo path, its class label, and its original annotation path.
        csv_rows.append([
            f"{label}/{image_file.name}",
            label,
            annotation_file.relative_to(DATASET_FOLDER),
        ])

    # Open labels.csv for writing.
    # "w" creates it or replaces an existing file.
    # newline="" is used to prevent extra blank lines in the CSV file.
    # encoding="utf-8" sets the text encoding.
    with (OUTPUT_FOLDER / "labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:

        # Create a CSV writer connected to the open file.
        writer = csv.writer(csv_file)

        # Write the column names as the first row.
        writer.writerow(["image_path", "label", "source_annotation"])

        # Write all of the image records collected above.
        writer.writerows(csv_rows)

    # Print a summary to the terminal after the files are created.
    print("Dataset created:", OUTPUT_FOLDER)
    print("Images per class:", IMAGES_PER_CLASS)
    print("Total images:", len(selected))
    print("Labels saved to:", OUTPUT_FOLDER / "labels.csv")


# Run the main function when the script is executed directly.
if __name__ == "__main__":
    main()