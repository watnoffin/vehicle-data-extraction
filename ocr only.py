import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import imutils
import pytesseract
import numpy as np

# --- Configuration ---
# path to the Tesseract executable
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
except Exception as e:
    st.error(f"Pytesseract configuration failed. Check your Tesseract path: {e}")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO(r'C:/Users/Fatin/OneDrive/Documents/car/models/license-plate-finetune-v1s.pt')

model = load_model()

# --- Core Recognition Function ---
def recognize_license_plate(image):
    
    # 1. Run YOLO detection
    results = model.predict(source=image, conf=0.4, verbose=False)
    
    license_plate_image = None
    
    # Extract the detected plate image
    for r in results:
        boxes = r.boxes.xyxy
        if len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            # Use the original image data from the YOLO results object
            license_plate_image = r.orig_img[y1:y2, x1:x2]
            break
            
    if license_plate_image is None:
        return None, "No license plate detected by YOLOv8.", None

    # 2. Image Pre-processing for OCR (Resize, Grayscale, Threshold)
    
    # Resize for consistent processing
    resized_plate = imutils.resize(license_plate_image.copy(), width=500)
    final_image = resized_plate.copy()
    
    # Convert to Grayscale
    gray = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
    
    # Use the full resized image as the ROI
    ROI = resized_plate.copy()

    # 3. THRESHOLDING (Creates the clean OCR Input Image)
    gray_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    normalized_gray_roi = cv2.normalize(gray_roi, None, alpha=0, beta=255, 
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Inverted Otsu's Thresholding
    _, thresh = cv2.threshold(
        normalized_gray_roi, 
        0, 
        255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. FINAL OCR (Using the reliable PSM 6 method)
    ocr_config = r'--psm 6' 
    raw_plate_text = pytesseract.image_to_string(thresh, config=ocr_config)
    
    # Clean the result: remove spaces and convert to uppercase
    final_plate_text = raw_plate_text.strip().replace(" ", "").upper()
    
    # --- Prepare Final Output Image (Green Box) ---
    w_roi, h_roi = final_image.shape[1], final_image.shape[0]
    
    # Draw green box
    cv2.rectangle(final_image, (0, 0), (w_roi - 1, h_roi - 1), (0, 255, 0), 3)
    
    if final_plate_text:
        # Overlay the recognized text (Blue text, top left)
        cv2.putText(final_image, final_plate_text, (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert BGR to RGB for Streamlit display
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        
        # Create the OCR Input Image for display
        ocr_input_image_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
        return final_plate_text, "Success", final_image_rgb, ocr_input_image_rgb
        
    return None, "Plate detected, but no characters recognized.", None, None


# --- Streamlit App Interface ---

st.title("Streamlit License Plate Recognizer")
st.markdown("Upload an image containing a car's license plate to run the YOLOv8 detection and OCR extraction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file as bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
    st.markdown("---")
    
    # Run recognition when button is clicked
    if st.button('Run OCR Recognition'):
        with st.spinner('Detecting and Extracting License Plate...'):
            plate_text, status, result_image, ocr_input_image = recognize_license_plate(img)
            
            st.subheader("Recognition Results")
            
            if status == "Success":
                # Display the OCR Input Image for debugging/visualization
                st.markdown("### Visualization of OCR Input")
                st.image(ocr_input_image, caption='Cleaned Input for Tesseract (PSM 6)', use_container_width=True)
                
                st.success(f"License Plate Number: **{plate_text}**")
                
                # Display the final image
                st.image(result_image, caption=f'Detected Plate: {plate_text}', use_container_width=True)
                
            elif status == "No license plate detected by YOLOv8.":
                st.error(status)
            else:
                st.warning(status)