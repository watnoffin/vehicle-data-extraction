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
import pymongo
from gridfs import GridFSBucket
import datetime

# --- GEMINI & MONGODB Initialization (Unchanged) ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Gemini API key not found. Please set 'GEMINI_API_KEY' in .streamlit/secrets.toml.")
    st.stop()

# --- MongoDB Configuration (Cached) ---
@st.cache_resource
def init_mongodb():
    try:
        mongo_uri = st.secrets["MONGO_URI"] 
        client = pymongo.MongoClient(mongo_uri)
        db = client.lpr_db 
        fs = GridFSBucket(db) 
        return client, db, fs
    except Exception as e:
        st.error(f"⚠️ MongoDB connection error. Please check MONGO_URI secret: {e}")
        st.stop()

mongo_client, mongo_db, fs_bucket = init_mongodb()
data_collection = mongo_db.vehicle_data 

# --- TESSERACT & YOLO CONFIGURATION ---
try:
    # pytesseract.pytesseract.tesseract_cmd = r'models/tesseract.exe'
    # pass
    # Set the path to the standard Tesseract executable location on Linux
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except Exception:
    pass

@st.cache_resource
def load_yolo_model():
    return YOLO(r'model/license-plate-finetune-v1s.pt')

yolo_model = load_yolo_model()

# --- GEMINI Helper Function ---
def analyze_vehicle_specifications(image_bytes: bytes, mime_type: str) -> dict:
    """
    Analyzes a single vehicle image using the Gemini API and returns
    its specifications as a dictionary.
    """
    
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
                        "mime_type": mime_type, 
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

        parsed_data = json.loads(response.text)
        
        if isinstance(parsed_data, list):
             return parsed_data[0]
        return parsed_data

    except Exception as e:
        st.error(f"⚠️ Gemini API Error: Could not extract vehicle specifications. {e}")
        return {
            "type": "Error", "make": "Error", "model": "Error",
            "year": "Error", "color": "Error"
        }

# --- OCR/ANPR Function (Unchanged) ---
# ... (Keep the existing recognize_license_plate function) ...
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
        return None, "No license plate detected by YOLOv8.", None

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
st.set_page_config(page_title="Vehicle Plate Recognitio n& Specification", layout="wide")
st.title("License Plate Recognition & Vehicle Specification Extraction")
st.markdown("Upload **car images**. We will process each image sequentially.")
st.markdown("---")

# *** Enable Multiple Files ***
uploaded_files = st.file_uploader("Upload Image(s) (JPG or PNG)", 
                                  type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

if uploaded_files:

    if st.button(f'Analyze {len(uploaded_files)} Vehicle(s)', type="primary"):
        st.session_state['run_analysis'] = True
    else:
        st.session_state['run_analysis'] = False
        
    st.markdown("---")

    if st.session_state.get('run_analysis', False):
        
        total_files = len(uploaded_files)
        st.success(f"Starting analysis for **{total_files}** image(s).")
        
        # Lists to hold records for final bulk insert (though we insert one by one here)
        all_vehicle_records = []
        
        # *** Loop through all uploaded files ***
        for i, uploaded_file in enumerate(uploaded_files):
            
            file_name = uploaded_file.name
            st.header(f"Processing Image {i+1} of {total_files}: {file_name}")
            
            try:
                # 1. Read file bytes
                file_bytes = uploaded_file.getvalue()
                
                # 2. Convert bytes to OpenCV image for YOLO/OCR
                nparr = np.frombuffer(file_bytes, np.uint8)
                img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # --- A. OCR/ANPR (License Plate) ---
                with st.spinner(f'({i+1}/{total_files}) 1. Extracting License Plate Number...'):
                    plate_text, plate_status, result_image_rgb = recognize_license_plate(img_cv2)
                
                # --- B. GEMINI VISION (Specifications) ---
                with st.spinner(f'({i+1}/{total_files}) 2. Analyzing Vehicle Specifications...'):
                    specs = analyze_vehicle_specifications(file_bytes, uploaded_file.type)

                # Display Results in columns for the current image
                col_img, col_data = st.columns([1, 1])
                
                with col_img:
                    st.image(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), caption=f"Original: {file_name}", width='stretch')
                    if result_image_rgb is not None:
                        st.image(result_image_rgb, caption='Detected Plate', width='stretch')

                with col_data:
                    st.subheader("Extracted Data")
                    st.markdown(f"**License Plate:** {plate_text if plate_status == 'Success' else 'NOT RECOGNIZED'} ({plate_status})")
                    st.markdown("---")
                    st.markdown(f"**Type:** {specs.get('type', 'N/A').title()}")
                    st.markdown(f"**Make:** {specs.get('make', 'N/A').title()}")
                    st.markdown(f"**Model:** {specs.get('model', 'N/A').title()}")
                    st.markdown(f"**Year:** {specs.get('year', 'N/A')}")
                    st.markdown(f"**Color:** {specs.get('color', 'N/A').title()}")


                # --- C. MongoDB (Storage) ---
                with st.spinner(f'({i+1}/{total_files}) 3. Storing Data in MongoDB...'):
                    
                    # Store image in GridFS
                    file_id = fs_bucket.upload_from_stream(
                        filename=file_name,
                        source=BytesIO(file_bytes), 
                        metadata={
                            "contentType": uploaded_file.type,
                            "upload_date": datetime.datetime.utcnow(),
                            "processed_by": "ANPR_Gemini_App"
                        }
                    )
                    
                    # Create the structured document
                    vehicle_record = {
                        "timestamp": datetime.datetime.utcnow(),
                        "license_plate": plate_text if plate_status == "Success" else "NOT_RECOGNIZED",
                        "specifications": specs,
                        "image_file_id": file_id, # Link to the file in GridFS
                        "original_filename": file_name,
                    }
                    
                    # Insert the structured document (Sequential inserts)
                    insert_result = data_collection.insert_one(vehicle_record)
                    
                st.info(f"✅ Data for **{file_name}** successfully stored. DB ID: {insert_result.inserted_id}")
                
            except Exception as e:
                st.error(f"❌ Critical Error processing {file_name}: {e}")
                
            st.markdown("---")

        st.balloons()
        st.success(f"Batch processing complete! Successfully analyzed and stored {total_files} records.")
        
    else:
        st.info(f"Click the button above to begin analysis for the {len(uploaded_files)} uploaded image(s).")
else:
    st.info("Please upload one or more images to begin.")

# --- Footer Section ---
st.markdown("---")

footer_html = """
<div style="text-align: center; color: grey; font-size: 0.8em; padding: 10px 0;">
    &copy; 2025 Butter Idea Sdn Bhd. All rights reserved. 
    <br> 
    Powered by <a href="https://www.bi2u.com/" target="_blank" style="color: grey; text-decoration: underline;">bi2u</a>.
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)