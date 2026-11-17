
"""
NEURONOVA - AI VISION CHALLENGE
Object detection, colour, shape, orientation, location and counting.

Install:
pip install ultralytics opencv-python numpy

Run:
python NeuroNova_Vision.py
"""

import cv2
import time
import os
import argparse
import numpy as np
from ultralytics import YOLO

TARGET_CLASSES = [
    "person", "shoe", "mobile phone", "bottle",
    "laptop", "chair", "book", "table",
    "keyboard", "backpack", "cup", "pen"
]

ALIASES = {
    "shoes": "shoe",
    "phone": "mobile phone",
    "cell phone": "mobile phone",
    "mug": "cup"
}


def normalise_name(name):
    name = " ".join(name.lower().strip().split())
    return ALIASES.get(name, name)


def detect_colour(crop):
    if crop is None or crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    chromatic = pixels[
        (pixels[:, 1] > 45) & (pixels[:, 2] > 35)
    ]

    if len(chromatic) > max(20, int(len(pixels) * 0.04)):
        hue = float(np.median(chromatic[:, 0]))

        if hue < 8 or hue >= 170:
            return "Red"
        elif hue < 20:
            return "Orange"
        elif hue < 34:
            return "Yellow"
        elif hue < 82:
            return "Green"
        elif hue < 100:
            return "Cyan"
        elif hue < 132:
            return "Blue"
        elif hue < 155:
            return "Purple"
        else:
            return "Pink"

    value = float(np.median(pixels[:, 2]))
    saturation = float(np.median(pixels[:, 1]))

    if value < 48:
        return "Black"
    elif value > 205 and saturation < 45:
        return "White"
    elif saturation < 55:
        return "Gray"
    return "Brown"


def detect_shape_and_orientation(crop):
    if crop is None or crop.size == 0:
        return "Unknown", "Unknown"

    height, width = crop.shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 45, 130)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    shape = "Unknown"
    orientation = "Undetermined"

    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0 and area > 30:
            approximation = cv2.approxPolyDP(
                contour, 0.035 * perimeter, True
            )

            vertices = len(approximation)
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > 0.78:
                shape = "Circle-like"
            elif vertices == 3:
                shape = "Triangle-like"
            elif vertices == 4:
                shape = "Rectangle / Square"
            elif vertices == 5:
                shape = "Pentagon-like"
            elif vertices == 6:
                shape = "Hexagon-like"
            else:
                shape = "Irregular"

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]

            if min(rw, rh) > 0:
                ratio = max(rw, rh) / min(rw, rh)

                if ratio < 1.18:
                    orientation = "Near-square / Face-on"
                elif rw > rh:
                    orientation = "Horizontal"
                else:
                    orientation = "Vertical"

    if orientation == "Undetermined":
        ratio = width / max(height, 1)

        if ratio > 1.25:
            orientation = "Horizontal"
        elif ratio < 0.80:
            orientation = "Vertical"
        else:
            orientation = "Near-square"

    return shape, orientation


def choose_reference():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        path = filedialog.askopenfilename(
            title="Select target reference image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        root.destroy()
        return path

    except Exception as error:
        print("Reference selector error:", error)
        return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["auto", "name", "reference"],
                        default="auto")
    parser.add_argument("--target", default="bottle")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--detect-every", type=int, default=1)

    args = parser.parse_args()

    print("\nNEURONOVA INITIALISING...")

    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(TARGET_CLASSES)

    print("AI MODEL LOADED SUCCESSFULLY")

    mode = args.mode
    target = normalise_name(args.target)

    reference = None

    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera unavailable. Try --camera 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_number = 0
    cached_results = []
    detector_fps = 0.0
    display_fps = 0.0
    last_fps_time = time.perf_counter()
    fps_frames = 0

    print("\nNEURONOVA READY")
    print("Q = Quit")
    print("M = Change mode")
    print("T = Enter target name")
    print("R = Select reference image")
    print("S = Save screenshot")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame_number += 1
            fps_frames += 1

            height, width = frame.shape[:2]

            if (
                frame_number % max(1, args.detect_every) == 0
                or not cached_results
            ):

                start = time.perf_counter()

                results = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=0.50,
                    max_det=30,
                    device="cpu",
                    verbose=False
                )[0]

                elapsed = max(
                    time.perf_counter() - start, 0.000001
                )

                detector_fps = 1.0 / elapsed

                cached_results = []

                if results.boxes is not None:

                    for box in results.boxes:

                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        name = normalise_name(
                            results.names[class_id]
                        )

                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].cpu().numpy()
                        )

                        x1 = max(0, min(width - 1, x1))
                        y1 = max(0, min(height - 1, y1))
                        x2 = max(x1 + 1, min(width, x2))
                        y2 = max(y1 + 1, min(height, y2))

                        crop = frame[y1:y2, x1:x2]

                        colour = detect_colour(crop)

                        shape, orientation = (
                            detect_shape_and_orientation(crop)
                        )

                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        cached_results.append({
                            "name": name,
                            "confidence": confidence,
                            "box": (x1, y1, x2, y2),
                            "colour": colour,
                            "shape": shape,
                            "orientation": orientation,
                            "location": (
                                f"X:{cx / width:.2f}, "
                                f"Y:{cy / height:.2f}"
                            )
                        })

            visible = []

            for item in cached_results:

                if mode == "name" and item["name"] != target:
                    continue

                visible.append(item)

            display = frame.copy()

            for index, item in enumerate(visible):

                x1, y1, x2, y2 = item["box"]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{item['name']} "
                    f"{item['confidence']:.2f}"
                )

                cv2.putText(
                    display,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            now = time.perf_counter()

            if now - last_fps_time >= 1.0:

                display_fps = fps_frames / (
                    now - last_fps_time
                )

                fps_frames = 0
                last_fps_time = now

            cv2.putText(
                display,
                f"NEURONOVA | MODE: {mode.upper()}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Display FPS: {display_fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Detector FPS: {detector_fps:.1f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Objects: {len(visible)}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            panel_y = 125

            for index, item in enumerate(visible[:6], 1):

                report = (
                    f"{index}. {item['name']} | "
                    f"{item['colour']} | "
                    f"{item['shape']} | "
                    f"{item['orientation']}"
                )

                cv2.putText(
                    display,
                    report[:100],
                    (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )

                panel_y += 20

            cv2.imshow(
                "NEURONOVA - AI VISION CHALLENGE",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("m"):

                modes = ["auto", "name", "reference"]

                mode = modes[
                    (modes.index(mode) + 1) % len(modes)
                ]

                if mode == "reference":
                    path = choose_reference()

                    if path:
                        reference = cv2.imread(path)
                        print("Reference image loaded.")
                    else:
                        mode = "auto"

                print("Mode:", mode)

            elif key == ord("t"):

                cv2.destroyAllWindows()

                entered = input(
                    "Enter target name: "
                ).strip()

                if entered:
                    target = normalise_name(entered)
                    mode = "name"

            elif key == ord("r"):

                path = choose_reference()

                if path:
                    reference = cv2.imread(path)
                    mode = "reference"
                    print("Reference loaded.")

            elif key == ord("s"):

                filename = "NeuroNova_snapshot.jpg"

                cv2.imwrite(filename, display)

                print("Saved:", filename)

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("\nNEURONOVA STOPPED")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as error:
        print("\nNEURONOVA ERROR:", error)