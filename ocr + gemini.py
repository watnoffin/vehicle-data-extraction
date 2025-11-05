import streamlit as st
from ultralytics import YOLO
import cv2
import imutils
import pytesseract
import numpy as np
import json
import base64
import re
from io import BytesIO
from google import genai
from google.genai.types import Part
from PIL import Image as PILImage 

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Gemini API key not found. Please set 'GEMINI_API_KEY' in .streamlit/secrets.toml.")
    st.stop()


# --- TESSERACT & YOLO CONFIGURATION ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
except Exception:
    pass

@st.cache_resource
def load_yolo_model():
    return YOLO(r'C:/Users/Fatin/OneDrive/Documents/car/models/license-plate-finetune-v1s.pt')

yolo_model = load_yolo_model()

# --- GEMINI Helper Function (Adapted for Streamlit Upload) ---

def analyze_vehicle_specifications(image_bytes: bytes, mime_type: str) -> dict:
    """
    Analyzes a single vehicle image using the Gemini API and returns
    its specifications as a dictionary.
    """
    
    # 1. Encode image bytes to Base64
    base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": (
                    "Examine the main vehicle in this image. "
                    "Respond ONLY in JSON with keys: "
                        "- type: choose one: car, truck, motorcycle, bus. If uncertain, use 'vehicle'. "
                        "- make: e.g., Toyota, Honda, Ford "
                        "- model: model name only, no trims/generations "
                        "- year: single closest model year (e.g. 2019) "
                        "- color: main visible color "
                )},
                {
                    "inline_data": {
                        "mime_type": mime_type, # Use the MIME type from the uploader
                        "data": base64_encoded_image
                    }
                }
            ]
        }
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config={"response_mime_type": "application/json"}
        )

        # The response text is a JSON string, load it into a Python dict
        parsed_data = json.loads(response.text)
        
        # Ensure the output is a dictionary (handle potential list output from batch logic)
        if isinstance(parsed_data, list):
             return parsed_data[0]
        return parsed_data

    except Exception as e:
        st.error(f"⚠️ Gemini API Error: Could not extract vehicle specifications. {e}")
        return {
            "type": "Error", "make": "Error", "model": "Error",
            "year": "Error", "color": "Error"
        }

# --- OCR/ANPR Function (License Plate) ---
def recognize_license_plate(image):
    
    # 1. Run YOLO detection
    results = yolo_model.predict(source=image, conf=0.4, verbose=False)
    
    license_plate_image = None
    
    # Extract the detected plate image
    for r in results:
        boxes = r.boxes.xyxy
        if len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            license_plate_image = r.orig_img[y1:y2, x1:x2]
            break
            
    if license_plate_image is None:
        return None, "No license plate detected by YOLOv8.", None, None

    # 2. Image Pre-processing for OCR (Resize, Grayscale, Threshold)
    resized_plate = imutils.resize(license_plate_image.copy(), width=500)
    final_image = resized_plate.copy()
    
    # 3. THRESHOLDING (OCR Input Image)
    gray_roi = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
    normalized_gray_roi = cv2.normalize(gray_roi, None, alpha=0, beta=255, 
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, thresh = cv2.threshold(
        normalized_gray_roi, 
        0, 
        255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. FINAL OCR (Using the reliable PSM 6 method)
    ocr_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_plate_text = pytesseract.image_to_string(thresh, config=ocr_config)
    
    #removes non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_plate_text)
    final_plate_text = cleaned_text.strip().replace(" ", "").upper()
    
    # --- Prepare Final Output Image (Green Box) ---
    w_roi, h_roi = final_image.shape[1], final_image.shape[0]
    cv2.rectangle(final_image, (0, 0), (w_roi - 1, h_roi - 1), (0, 255, 0), 3)
    
    if final_plate_text:
        cv2.putText(final_image, final_plate_text, (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        return final_plate_text, "Success", final_image_rgb
        
    return None, "Plate detected, but no characters recognized.", None


# --- Streamlit App Interface ---

st.set_page_config(page_title="🚗 ANPR + Vehicle Specifications", layout="wide")
st.title("License Plate Recognition & Vehicle Specification Extraction")
st.markdown("Upload an image of a car. We will extract the **License Plate Number** (using YOLO+OCR) and the **Vehicle Specifications** (using Gemini's vision model).")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read the image file as bytes
    file_bytes = uploaded_file.getvalue()
    
    # 2. Convert bytes to OpenCV image for OCR/YOLO
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Create columns for display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), caption=uploaded_file.name, width='stretch')
        
        if st.button('Analyze Vehicle', type="primary"):
            st.session_state['run_analysis'] = True
        else:
            st.session_state['run_analysis'] = False
            
    with col2:
        if st.session_state.get('run_analysis', False):
            # --- START ANALYSIS ---
            
            # --- A. OCR/ANPR (License Plate) ---
            with st.spinner('1. Extracting License Plate Number (YOLO + OCR)...'):
                plate_text, plate_status, result_image_rgb = recognize_license_plate(img_cv2)

            st.subheader("📝 License Plate Extraction")
            if plate_status == "Success":
                st.success(f"Plate Number: **{plate_text}**")
                st.image(result_image_rgb, caption='Detected Plate with Bounding Box', width='stretch')
                
            else:
                st.error(f"License Plate Error: {plate_status}")
                
            st.markdown("---")
            
            # --- B. GEMINI VISION (Specifications) ---
            with st.spinner('2. Analyzing Vehicle Specifications (Gemini 2.5 Flash)...'):
                # Pass the original image bytes and mime type to Gemini function
                specs = analyze_vehicle_specifications(file_bytes, uploaded_file.type)

            st.subheader("🤖 Vehicle Specification Extraction")
            
            # Use columns to display the specs neatly
            spec_col1, spec_col2 = st.columns(2)
            
            # Display specs in a clean, formatted way
            spec_col1.markdown(f"**Type:** {specs.get('type', 'N/A').title()}")
            spec_col2.markdown(f"**Color:** {specs.get('color', 'N/A').title()}")
            spec_col1.markdown(f"**Make:** {specs.get('make', 'N/A').title()}")
            spec_col2.markdown(f"**Year:** {specs.get('year', 'N/A')}")
            st.markdown(f"**Model:** {specs.get('model', 'N/A').title()}")

        else:
            st.info("Click 'Analyze Vehicle' to begin the process.")