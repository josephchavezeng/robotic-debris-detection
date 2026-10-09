# Run a Streamlit dashboard for the simulated debris-inspection robot.
#
# The dashboard loads the trained ResNet-50 checkpoint and the saved
# held-out image split. A scan button runs one image through the model.
# A debris prediction pauses the simulated robot and logs the scan;
# a no-debris prediction moves it forward.
#
# This dashboard simulates robot behavior only. It does not send commands
# to physical robot hardware.
#
# Start it from the project folder with:
# source .venv/bin/activate
# python -m streamlit run Debris_Robot/debris_robot_dashboard.py
# python -m streamlit run debris_robot_dashboard.py

# json reads the saved evaluation metrics file.
import json

# datetime adds a timestamp to each robot event in the event log.
from datetime import datetime

# Path builds file and folder locations relative to this script.
from pathlib import Path

# pandas reads CSV files and displays event logs as tables.
import pandas as pd

# Image opens the camera photos from the dataset.
from PIL import Image

# Streamlit creates the dashboard interface and manages interactive state.
import streamlit as st

# components embeds the robot drawing inside the Streamlit page.
import streamlit.components.v1 as components

# PyTorch loads the trained model and runs image predictions.
import torch
import torch.nn as nn

# transforms prepare images in the same way they were prepared for training.
from torchvision import transforms

# resnet50 builds the model architecture used during training.
from torchvision.models import resnet50


# Find the folder containing this dashboard script.
BASE_DIR = Path(__file__).resolve().parent

# Locate the prepared dataset and saved model outputs beside the script.
DATA_DIR = BASE_DIR / "custom_debris_dataset_420_each"
OUTPUT_DIR = BASE_DIR / "outputs"

# Locate the trained model checkpoint, saved train/test split,
# and evaluation metrics.
CHECKPOINT_PATH = OUTPUT_DIR / "models" / "resnet50_debris.pth"
SPLIT_PATH = OUTPUT_DIR / "split_manifest.csv"
METRICS_PATH = OUTPUT_DIR / "evaluation" / "test_metrics.json"

# Map class names to numbers and numbers back to class names.
CLASS_TO_IDX = {"no_solid_debris": 0, "solid_debris": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}


# Set the browser tab title and wide page layout.
st.set_page_config(
    page_title="Debris Inspection Robot",
    layout="wide",
)


def choose_device():
    # Use an NVIDIA GPU when CUDA is available.
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Use Apple's GPU support on a Mac when MPS is available.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Otherwise, run the model on the CPU.
    return torch.device("cpu")


# Cache the loaded model so Streamlit button clicks do not reload it each time.
@st.cache_resource
def load_model(checkpoint_path):
    # Select the available device.
    device = choose_device()

    # Load the saved checkpoint onto the CPU first.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Rebuild the same ResNet-50 structure used in training.
    # weights=None prevents loading ImageNet weights again; the trained
    # weights are restored from the checkpoint below.
    model = resnet50(weights=None)

    # Replace ResNet-50's original output layer with one debris-classification score.
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Load the trained weights into the model.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the selected device and set it to evaluation mode.
    model.to(device)
    model.eval()

    # Read the image size and normalization settings saved during training.
    # Defaults are used if a setting is missing from the checkpoint.
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("normalization_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization_std", [0.229, 0.224, 0.225])

    # Prepare the same resize, tensor conversion, and normalization steps
    # used for evaluation during model training.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Read the probability cutoff used to choose between the two classes.
    threshold = float(checkpoint.get("threshold", 0.5))

    # Return everything needed to make predictions.
    return model, device, transform, threshold


# Cache the test split table because it does not change between button clicks.
@st.cache_data
def load_test_images(split_path):
    # Read the CSV containing the saved train/test assignments.
    split_frame = pd.read_csv(split_path)

    # Keep only rows marked as held-out test images.
    test_frame = split_frame[split_frame["split"] == "test"].copy()

    # Stop if the split file has no held-out images.
    if test_frame.empty:
        raise ValueError("The saved split manifest has no test images.")

    # Reset row numbers so the dashboard can move through images by index.
    return test_frame.reset_index(drop=True)


def predict_image(image_path, model, device, transform, threshold):
    # Open the image in a context manager so the file closes after reading.
    with Image.open(image_path) as source:
        # Convert the photo to RGB to ensure it has three color channels.
        image = source.convert("RGB")

    # Apply preprocessing, add a batch dimension, and move it to the model's device.
    tensor = transform(image).unsqueeze(0).to(device)

    # Run prediction without calculating gradients or updating model weights.
    with torch.inference_mode():
        # Get the model's raw score logit.
        logit = model(tensor)

        # Convert the logit to a probability between 0 and 1.
        debris_probability = torch.sigmoid(logit).item()

    # Convert probability into class 1 or 0 using the configured threshold.
    predicted_idx = int(debris_probability >= threshold)

    # Convert the predicted number back to a readable class name.
    predicted_label = IDX_TO_CLASS[predicted_idx]

    # Calculate the probability of the predicted class.
    # For debris, use p; for no debris, use 1 - p.
    confidence = (
        debris_probability
        if predicted_idx == 1
        else 1.0 - debris_probability
    )

    # Return the class name, debris probability, and predicted-class score.
    return predicted_label, debris_probability, confidence


def add_log(event):
    # Create an empty event list if this is the first recorded event.
    if "event_log" not in st.session_state:
        st.session_state.event_log = []

    # Add the current local time and supplied event details to the log.
    st.session_state.event_log.append({
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        **event,
    })


def initialize_state():
    # Set up the dashboard's starting values.
    defaults = {
        "status": "READY",
        "scan_index": 0,
        "robot_position": 0,
        "last_scan": None,
        "event_log": [],
    }

    # Add each starting value only if it is not already in Streamlit's
    # session state. This preserves progress when the page reruns.
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def robot_svg(position):
    # Move the robot proportionally across the lane over 168 scans.
    # Keep the robot within the visible drawing area.
    robot_x = 55 + min(position, 168) * (335 / 168)

    # Build an SVG drawing of the factory floor and simulated robot.
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 230" role="img"
         aria-label="Top-down simulated wheeled debris inspection robot">

      <!-- Draw the background with rounded corners.
           Width and height define the drawing size, and rx rounds the corners. -->
      <rect width="680" height="230" rx="18" fill="#f8fafc"/>

      <!-- Draw the factory floor lane.
           x and y position the lane, width and height define its dimensions,
           and rx controls the rounded corners. -->
      <rect x="24" y="42" width="632" height="145" rx="16" fill="#e2e8f0"/>

      <!-- Draw the dashed centerline representing the inspection path.
           x1 and y1 define the start, x2 and y2 define the endpoint.
           stroke sets the color, stroke-width sets the thickness,
           and stroke-dasharray creates the dashed pattern. -->
      <line x1="36" y1="114" x2="644" y2="114" stroke="#94a3b8"
            stroke-width="3" stroke-dasharray="14 12"/>

      <!-- Display the factory floor label.
           x and y position the text, while font-family, font-size,
           and fill control its appearance. -->
      <text x="34" y="29" font-family="Arial" font-size="14" fill="#475569">
        FACTORY FLOOR · FORWARD SCAN LANE
      </text>

      <!-- Group the robot elements so they move together.
           translate uses robot_x to change the horizontal position
           while keeping the vertical position fixed at 83. -->
      <g transform="translate({robot_x} 83)">

        <!-- Draw the main robot body.
             fill sets the body color and stroke sets its outline. -->
        <rect x="0" y="4" width="93" height="60" rx="15" fill="#2563eb"
              stroke="#1e3a8a" stroke-width="3"/>

        <!-- Draw the four wheels.
             cx and cy position each wheel's center, and r sets its radius. -->
        <circle cx="18" cy="67" r="9" fill="#0f172a"/>
        <circle cx="75" cy="67" r="9" fill="#0f172a"/>
        <circle cx="18" cy="1" r="9" fill="#0f172a"/>
        <circle cx="75" cy="1" r="9" fill="#0f172a"/>

        <!-- Draw the arrow indicating the robot's forward direction.
             The path coordinates define the arrow's shaft and head. -->
        <path d="M91 34 L112 34 M105 26 L113 34 L105 42" fill="none"
              stroke="#1e3a8a" stroke-width="4" stroke-linecap="round"
              stroke-linejoin="round"/>

      </g>

    </svg>
    """

    # Return the SVG drawing for display in the Streamlit dashboard.
    return svg


def main():
    # Restore or initialize the robot's state for this browser session.
    initialize_state()

    # Display the dashboard title and explain that the robot is simulated.
    st.title("Factory Floor Debris Inspection")
    st.caption(
        "ResNet-50 image classifier connected to a simulated wheeled robot. "
        "This demo sends no commands to physical hardware."
    )

    # Load the held-out image list; show an error if loading fails.
    try:
        test_frame = load_test_images(str(SPLIT_PATH))
    except Exception as error:
        st.error(f"Could not load the held-out test split: {error}")
        st.stop()

    # Check whether any image path in the test split points to a missing file.
    missing_images = [
        DATA_DIR / image_path
        for image_path in test_frame["image_path"]
        if not (DATA_DIR / image_path).is_file()
    ]

    # Stop if a test image is missing, and show the first missing path.
    if missing_images:
        st.error(f"Test image file is missing: {missing_images[0]}")
        st.stop()

    # Load the model once; show an error if the checkpoint cannot be loaded.
    try:
        model, device, transform, threshold = load_model(str(CHECKPOINT_PATH))
    except Exception as error:
        st.error(f"Could not load the trained ResNet-50 checkpoint: {error}")
        st.stop()

    # Mark the robot run complete when all held-out images have been scanned.
    if st.session_state.scan_index >= len(test_frame):
        st.session_state.status = "COMPLETE"

    # Build the sidebar and show the run configuration.
    sidebar = st.sidebar
    sidebar.header("Run summary")
    sidebar.write(f"**Device:** {device}")
    sidebar.write(f"**Held-out images:** {len(test_frame)}")
    sidebar.write(f"**Decision threshold:** {threshold:.2f}")
    sidebar.write("**Robot mode:** simulated")

    # Display saved F1 and AUROC metrics from the evaluation file.
    if METRICS_PATH.is_file():
        try:
            # Read previously saved test metrics from the JSON file.
            with METRICS_PATH.open(encoding="utf-8") as file:
                metrics = json.load(file)

            # Show the two selected metrics in the sidebar.
            sidebar.metric("Test F1", f"{metrics['f1_solid_debris']:.3f}")
            sidebar.metric("Test AUROC", f"{metrics['auroc']:.3f}")

        # Show a message if the metrics file cannot be read or lacks these values.
        except (OSError, KeyError, ValueError):
            sidebar.caption("Saved evaluation metrics could not be read.")

    # Reset all demo state when the sidebar reset button is clicked.
    if sidebar.button("Reset demo", use_container_width=True):
        st.session_state.status = "READY"
        st.session_state.scan_index = 0
        st.session_state.robot_position = 0
        st.session_state.last_scan = None
        st.session_state.event_log = []

        # Rerun Streamlit so the reset state appears immediately.
        st.rerun()

    # Create two columns for the robot controls and status text.
    control_col1, control_col2 = st.columns([1, 2])

    # Place the interactive buttons in the left column.
    with control_col1:

        # In READY state, offer a button to begin inspection.
        if st.session_state.status == "READY":
            if st.button(
                "Start robot",
                type="primary",
                use_container_width=True,
            ):
                # Change the state to moving and record the start event.
                st.session_state.status = "MOVING"
                add_log({
                    "event": "START",
                    "action": "BEGIN_INSPECTION",
                })

                # Rerun so the scan button appears.
                st.rerun()

        # In MOVING state, offer a button to scan the next held-out image.
        elif st.session_state.status == "MOVING":
            if st.button(
                "Scan next image",
                type="primary",
                use_container_width=True,
            ):
                # Get the row for the next image in the held-out split.
                row = test_frame.iloc[st.session_state.scan_index]

                # Build the full path to that image.
                image_path = DATA_DIR / row["image_path"]

                # Run the image through the classifier.
                label, debris_probability, confidence = predict_image(
                    image_path,
                    model,
                    device,
                    transform,
                    threshold,
                )

                # Decide whether the simulated robot pauses or moves forward.
                detected = label == "solid_debris"
                action = "STOP_AND_LOG" if detected else "MOVE_FORWARD"

                # Store the result for display and the event log.
                scan_record = {
                    "image_path": row["image_path"],
                    "predicted_label": label,
                    "debris_probability": debris_probability,
                    "confidence": confidence,
                    "action": action,
                }
                st.session_state.last_scan = scan_record
                add_log({"event": "SCAN", **scan_record})

                # If debris is predicted, pause for operator review.
                if detected:
                    st.session_state.status = "PAUSED - DEBRIS"
                else:
                    # Otherwise, advance the robot and move to the next image.
                    st.session_state.robot_position += 1
                    st.session_state.scan_index += 1

                    # Mark the run complete when there are no images left.
                    if st.session_state.scan_index >= len(test_frame):
                        st.session_state.status = "COMPLETE"

                # Rerun so the updated robot and prediction are displayed.
                st.rerun()

        # In PAUSED state, offer a button to simulate clearing the debris.
        elif st.session_state.status == "PAUSED - DEBRIS":
            if st.button(
                "Clear and resume",
                type="primary",
                use_container_width=True,
            ):
                # Record the operator's decision to clear the alert.
                add_log({
                    "event": "OPERATOR_CLEAR",
                    "image_path": (
                        st.session_state.last_scan["image_path"]
                        if st.session_state.last_scan else ""
                    ),
                    "action": "RESUME_FORWARD",
                })

                # Advance the robot and image index after the operator clears.
                st.session_state.robot_position += 1
                st.session_state.scan_index += 1

                # Resume scanning, or mark the run complete if all images are done.
                st.session_state.status = (
                    "COMPLETE"
                    if st.session_state.scan_index >= len(test_frame)
                    else "MOVING"
                )

                # Rerun so the new state and next scan appear.
                st.rerun()

        # For any finished state, show the completion message.
        else:
            st.success("All held-out demo images have been scanned.")

    # Show the current robot state and action rule in the right column.
    with control_col2:
        st.write(f"**Robot state:** {st.session_state.status}")
        st.write(
            "Action policy: **scan → log debris and pause, "
            "or move forward**."
        )

    # Draw the simulated robot using SVG and embed it in the dashboard.
    st.subheader("Simulated robot")
    components.html(
        robot_svg(st.session_state.robot_position),

        # Give the simulation enough height to show the full graphic.
        height=450,

        # Disable scrolling inside the embedded simulation frame.
        scrolling=False,
    )

    # Create side-by-side columns for the camera image and prediction.
    left, right = st.columns([1.2, 1])

    # Use the left column to show the current or next held-out image.
    with left:
        st.subheader("Camera image")

        # If paused, continue showing the image that triggered the pause.
        if (
            st.session_state.status == "PAUSED - DEBRIS"
            and st.session_state.last_scan
        ):
            shown_path = DATA_DIR / st.session_state.last_scan["image_path"]
            caption = "Current scan · robot paused for operator review"

        # Otherwise, show the next image if any remain.
        elif st.session_state.scan_index < len(test_frame):
            row = test_frame.iloc[st.session_state.scan_index]
            shown_path = DATA_DIR / row["image_path"]
            caption = (
                f"Next held-out scan · image "
                f"{st.session_state.scan_index + 1} of {len(test_frame)}"
            )

        # If scanning is complete, show the last scanned image if available.
        elif st.session_state.last_scan:
            shown_path = DATA_DIR / st.session_state.last_scan["image_path"]
            caption = "Final scanned image"

        # Before the first scan, there is no camera image to display.
        else:
            shown_path = None
            caption = "No image selected"

        # Display the selected image and its caption.
        if shown_path is not None:
            st.image(
                str(shown_path),
                caption=caption,
                use_container_width=True,
            )

    # Use the right column to show the latest prediction.
    with right:
        st.subheader("Latest model result")

        # Display prediction details if a scan has been run.
        if st.session_state.last_scan:
            result = st.session_state.last_scan

            # Convert "solid_debris" into display text "Solid Debris".
            predicted = result["predicted_label"].replace("_", " ").title()

            # Show the predicted class and probability scores.
            st.metric("Prediction", predicted)
            st.metric(
                "Debris probability",
                f"{result['debris_probability']:.1%}",
            )
            st.metric(
                "Confidence in predicted class",
                f"{result['confidence']:.1%}",
            )

            # Explain the action associated with the predicted class.
            if result["predicted_label"] == "solid_debris":
                st.warning("Debris predicted — robot stopped and logged the image.")
            else:
                st.success("No solid debris predicted — robot moved forward.")

        # Before the first scan, prompt the operator to start the demo.
        else:
            st.info("Start the robot, then scan a held-out test image.")

    # Display the event log collected during the current dashboard session.
    st.subheader("Inspection event log")

    # If events exist, convert them to a table and show a download button.
    if st.session_state.event_log:
        log_frame = pd.DataFrame(st.session_state.event_log)
        st.dataframe(
            log_frame,
            use_container_width=True,
            hide_index=True,
        )

        # Convert the table to CSV text and offer it as a download.
        st.download_button(
            "Download event log CSV",
            data=log_frame.to_csv(index=False).encode("utf-8"),
            file_name="debris_robot_demo_log.csv",
            mime="text/csv",
        )

    # Show a note when no scan or operator events have been recorded.
    else:
        st.caption("No scan events yet.")


# Run the main function when the script is executed directly.
if __name__ == "__main__":
    main()