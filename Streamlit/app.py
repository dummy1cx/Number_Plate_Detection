import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import pytesseract

# -----------------------------------------------------------------------------------
# The codes are referenced from stramlit dosumentation page and multiple github repos
# All the referenced links and website are mentioned in the written report
# ------------------------------------------------------------------------------------


# Load best saved model
model = YOLO("best.pt")  

st.set_page_config(page_title="Number Plate Detection", layout="centered")
st.title("Vehicle Number Detection")
st.write("Upload an image (JPG/PNG) or video (MP4) to detect and read number plates.")

# Upload file in jpg png and mp4 file format
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # For image analysis
    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # model inference
        results = model(image_np)
        result = results[0]
        img_with_boxes = result.plot()

        st.image(img_with_boxes, caption="Detected Number Plates", use_column_width=True)

        st.subheader("OCR Results:")
        found_text = False

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image_np[y1:y2, x1:x2]

            # OCR readble file format
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 7').strip()

            if text:
                found_text = True
                st.write(f"Plate Text: `{text}`")

        if not found_text:
            st.warning("No readable number plate text found.")

    # for video analysis .mp4 file format
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.info("Processing video... please wait..... ")

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "output_with_boxes.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        all_texts = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            results = model(frame)
            result = results[0]
            frame_with_boxes = result.plot()

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 7').strip()

                if text:
                    all_texts.append((frame_num, text))
                    # Draw text on the video
                    cv2.putText(frame_with_boxes, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

            out.write(frame_with_boxes)

            stframe.image(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)

        cap.release()
        out.release()

        st.success("Video processing complete!")

        # Displaying OCR results
        if all_texts:
            st.subheader("Detected Plate Numbers:")
            for fnum, text in all_texts:
                st.write(f"Frame {fnum}: `{text}`")
        else:
            st.warning("No readable number plates detected in the video.")

        # This enables the download option
        with open(output_path, "rb") as f:
            st.download_button("Download Result Video", f, file_name="number_plate_output.mp4", mime="video/mp4")
