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

# --- CONSTANTS ---
MAX_IMAGE_DISPLAY_WIDTH = 500 # Max width for displaying images in Streamlit

def show_alert(message):
    """Injects JavaScript to display a standard browser alert popup."""
    alert_js = f"""
    <script>
    alert('{message}');
    </script>
    """
    # Use st.markdown to inject the script. Streamlit executes the script upon rendering.
    st.markdown(alert_js, unsafe_allow_html=True)
# --- END NEW HELPER FUNCTION ---

# --- GEMINI & MONGODB Initialization ---
try:
    # Safely access secrets to init client
    client = genai.Client(api_key=st.secrets.get("GEMINI_API_KEY"))
    if not client:
        # Note: In a real environment, you might fetch this from another env var or fail gracefully
        st.error("Gemini API key not found in secrets.toml.")
        st.stop()
except Exception as e:
    st.error(f"Gemini API key not found or error initializing client: {e}")
    st.stop()

# --- MongoDB Configuration (Cached) ---
@st.cache_resource
def init_mongodb():
    try:
        mongo_uri = st.secrets.get("MONGO_URI") 
        if not mongo_uri:
            st.error("MONGO_URI not found in secrets.toml.")
            st.stop()
            
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

# --- GEMINI Helper Function: Specification Extraction (Unchanged) ---
def analyze_vehicle_specifications(image_bytes: bytes, mime_type: str) -> dict:
    """Analyzes a single vehicle image using the Gemini API and returns its specifications as a dictionary."""
    
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

# --- OCR/ANPR Function (YOLO + TESSERACT) (Unchanged) ---
def recognize_license_plate_ocr(image):
    """Recognizes license plate using YOLO detection and Tesseract OCR."""
    
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
    
    # removes non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_plate_text)
    final_plate_text = cleaned_text.strip().replace(" ", "").upper()
    
    # --- Prepare Final Output Image (Green Box) ---
    w_roi, h_roi = final_image.shape[1], final_image.shape[0]
    cv2.rectangle(final_image, (0, 0), (w_roi - 1, h_roi - 1), (0, 255, 0), 3)
    
    if final_plate_text:
        cv2.putText(final_image, final_plate_text, (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        return final_plate_text, "Success (OCR)", final_image_rgb
        
    return None, "Plate detected, but no characters recognized (OCR).", None

# --- GEMINI Helper Function: Plate Extraction (Unchanged) ---
def recognize_license_plate_gemini(image_bytes: bytes, mime_type: str):
    """Analyzes a single vehicle image using the Gemini API to extract the license plate number."""
    base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": (
                    "Examine the main vehicle in this image and identify the license plate number. "
                    "Respond ONLY with the cleaned, uppercase license plate text. "
                    "If a plate is not clearly visible, respond with 'NOT_FOUND'. "
                    "Do not include any explanation or extra characters."
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
            contents=contents
        )
        
        raw_plate_text = response.text.strip().upper()
        
        # removes non-alphanumeric characters (including NOT_FOUND)
        cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_plate_text)
        
        if cleaned_text == "NOTFOUND" or not cleaned_text:
            return None, "No license plate recognized (Gemini)."
            
        return cleaned_text, "Success (Gemini)"

    except Exception as e:
        st.error(f"⚠️ Gemini API Error for Plate Recognition: {e}")
        return None, "Error during Gemini plate recognition."

# --- MongoDB Storage Function (Unchanged) ---
def save_data_to_mongodb(
        file_name, file_bytes, mime_type, 
        plate_text, plate_status, specs
    ):
    """Handles the storage of image and data to GridFS and the main collection."""
    
    try:
        # Store image in GridFS
        file_id = fs_bucket.upload_from_stream(
            filename=file_name,
            source=BytesIO(file_bytes), 
            metadata={
                "contentType": mime_type,
                "upload_date": datetime.datetime.utcnow(),
                "processed_by": "ANPR_Gemini_App"
            }
        )
        
        # Create the structured document
        vehicle_record = {
            "timestamp": datetime.datetime.utcnow(),
            "license_plate": plate_text if plate_status.startswith("Success") else "NOT_RECOGNIZED",
            "specifications": specs,
            "image_file_id": file_id, # Link to the file in GridFS
            "original_filename": file_name,
        }
        
        # Insert the structured document
        insert_result = data_collection.insert_one(vehicle_record)
        
        return True, insert_result.inserted_id
    
    except Exception as e:
        return False, str(e)

# --- Clear All Function ---
def clear_all_data():
    # Delete all keys that store user input or analysis results
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    if 'run_analysis' in st.session_state:
        del st.session_state['run_analysis']
        
    if 'uploader_key' in st.session_state:
        del st.session_state['uploader_key'] 
    
    # Force a rerun to clear the files and redraw the initial UI
    st.rerun()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Vehicle Plate Recognition & Specification", layout="wide")
st.title("License Plate Recognition & Vehicle Specification Extraction")

# --- CUSTOM CSS BLOCK for LARGER PLATE TEXT---
st.markdown("""
<style>
/* Target the element that contains the green license plate text (inline code block) */
.stMarkdown code {
    font-size: 1.5em !important; /* Increase the font size */
    font-weight: bold;
    padding: 0.2em 0.4em; /* Adjust padding for better look */
}
</style>
""", unsafe_allow_html=True)
# --- END CUSTOM CSS BLOCK ---

st.markdown("Upload **car images**. We will process each image sequentially.")
st.markdown("---")

# Initialize session state for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = []
    
# *** Enable Multiple Files ***
uploaded_files = st.file_uploader("Upload Image(s) (JPG or PNG)", 
                                  type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)

col_buttons_1, col_buttons_2, _ = st.columns([1, 1, 3])

if uploaded_files:
    # --- Analyze Button ---
    if col_buttons_1.button(f'Analyze {len(uploaded_files)} Vehicle(s)', 
                 type="primary", 
                 key="analyze_btn",
                 help="Start the initial analysis using the default YOLO/OCR for plate extraction and Gemini for specifications."):
        st.session_state['analysis_results'] = [] # Clear previous results
        st.session_state['run_analysis'] = True
    else:
        # Only set run_analysis to False if no initial run has occurred or if button wasn't clicked
        if not st.session_state.get('analysis_results'):
             st.session_state['run_analysis'] = False
             
# --- Clear All Button (Placed next to Analyze Button) ---
if col_buttons_2.button("🗑️ Clear All Data & Restart", 
                       key="clear_btn",
                       help="Clears all uploaded files, analysis results, and resets the page.",
                       type="secondary"):
    clear_all_data()

st.markdown("---")

if uploaded_files:
    if st.session_state.get('run_analysis', False) or st.session_state.get('analysis_results'):
        
        if st.session_state.get('run_analysis', False):
            st.session_state['run_analysis'] = False # Reset the flag after starting
            total_files = len(uploaded_files)
            st.session_state['analysis_results'] = [] # Prepare to store new results
            
            # --- START Initial Batch Status ---
            with st.spinner(f"Analyzing {total_files} image(s)... This may take a moment."):
                
                # *** Loop through all uploaded files for INITIAL ANALYSIS ***
                for i, uploaded_file in enumerate(uploaded_files):
                    
                    file_name = uploaded_file.name
                    
                    try:
                        # 1. Read file bytes
                        file_bytes = uploaded_file.getvalue()
                        mime_type = uploaded_file.type
                        
                        # 2. Convert bytes to OpenCV image for YOLO/OCR
                        nparr = np.frombuffer(file_bytes, np.uint8)
                        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # --- A. OCR/ANPR (License Plate) ---
                        plate_text, plate_status, result_image_rgb = recognize_license_plate_ocr(img_cv2)
                        
                        # --- B. GEMINI VISION (Specifications) ---
                        specs = analyze_vehicle_specifications(file_bytes, mime_type)
                            
                        # Store results in session state
                        st.session_state['analysis_results'].append({
                            'index': i,
                            'file_name': file_name,
                            'file_bytes': file_bytes, # Store bytes for later save/Gemini re-run
                            'mime_type': mime_type,
                            'img_cv2': img_cv2,
                            'plate_text': plate_text,
                            'plate_status': plate_status,
                            'result_image_rgb': result_image_rgb,
                            'specs': specs,
                            'saved': False # Track if saved to DB
                        })

                    except Exception as e:
                        st.error(f"❌ Critical Error processing {file_name}: {e}")
                        
            st.success(f"Analysis complete for {total_files} record(s).")
            # --- END Initial Batch Status ---
            
        # --- Display and Interactive Section (Runs after initial or reload) ---
        
        if st.session_state['analysis_results']:
            
            # --- MODIFIED HEADER ---
            st.header("Results of Extraction")
            st.markdown("Review the extracted data below. You may **re-run** the plate recognition or **save** the final record to MongoDB.")
            st.markdown("---")
            # --- END MODIFIED HEADER ---
            
            # --- Individual Records Loop ---
            for i, record in enumerate(st.session_state['analysis_results']):
                
                file_name = record['file_name']
                file_bytes = record['file_bytes']
                mime_type = record['mime_type']
                img_cv2 = record['img_cv2']
                
                st.subheader(f"Record {i+1}: {file_name}")
                
                # Use a unique key for the container for re-render stability
                container = st.container(border=True, key=f"record_container_{i}")
                
                with container:
                    col_img, col_data = st.columns([1, 1])
                    
                    with col_img:
                        # STANDARDIZE IMAGE DISPLAY SIZE
                        st.image(
                            cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), 
                            caption=f"Original: {file_name}", 
                            width=MAX_IMAGE_DISPLAY_WIDTH
                        )
                        if record['result_image_rgb'] is not None:
                            st.image(record['result_image_rgb'], caption=f"Detected Plate ({record['plate_status'].split(' ')[-1].strip('()')})", width='stretch')

                    with col_data:
                        st.markdown(f"**Plate Status:** {'✅ SAVED' if record['saved'] else '⚠️ PENDING SAVE'}")
                        st.markdown("---")
                        # Uses the CSS-enlarged inline code block
                        st.markdown(f"**Current License Plate:** **`{record['plate_text'] if record['plate_text'] else 'NOT RECOGNIZED'}`**") 
                        st.markdown(f"**Method:** *{record['plate_status']}*")
                        st.markdown("---")
                        st.markdown(f"**Type:** {record['specs'].get('type', 'N/A').title()}")
                        st.markdown(f"**Make:** {record['specs'].get('make', 'N/A').title()}")
                        st.markdown(f"**Model:** {record['specs'].get('model', 'N/A').title()}")
                        st.markdown(f"**Year:** {record['specs'].get('year', 'N/A')}")
                        st.markdown(f"**Color:** {record['specs'].get('color', 'N/A').title()}")

                        # --- Individual Interactive Buttons ---
                        st.markdown("---")
                        
                        col_actions_1, col_actions_2, col_actions_3 = st.columns(3)
                        
                        # Button 1: Re-run with YOLO/OCR
                        if col_actions_1.button("Re-run YOLO/OCR", 
                                                key=f"rerun_ocr_{i}", 
                                                help="Rerun the traditional computer vision (YOLO detection + Tesseract OCR) process for the license plate."):
                            with st.spinner(f'Re-running YOLO/OCR for {file_name}...'):
                                new_plate_text, new_plate_status, new_result_image_rgb = recognize_license_plate_ocr(img_cv2)
                                st.session_state['analysis_results'][i]['plate_text'] = new_plate_text
                                st.session_state['analysis_results'][i]['plate_status'] = new_plate_status
                                st.session_state['analysis_results'][i]['result_image_rgb'] = new_result_image_rgb
                                st.session_state['analysis_results'][i]['saved'] = False 
                                st.toast(f"YOLO/OCR re-run complete for {file_name}. New Plate: {new_plate_text if new_plate_text else 'NOT FOUND'}")
                                st.rerun()
                        
                        # Button 2: Run with Gemini
                        if col_actions_2.button("Run with Gemini", 
                                                key=f"rerun_gemini_{i}", 
                                                help="Use Google Gemini Vision to extract the license plate number. This may be more accurate for difficult images."):
                            with st.spinner(f'Running Gemini for plate extraction on {file_name}...'):
                                new_plate_text, new_plate_status = recognize_license_plate_gemini(file_bytes, mime_type)
                                
                                st.session_state['analysis_results'][i]['plate_text'] = new_plate_text
                                st.session_state['analysis_results'][i]['plate_status'] = new_plate_status
                                st.session_state['analysis_results'][i]['result_image_rgb'] = None 
                                st.session_state['analysis_results'][i]['saved'] = False 
                                st.toast(f"Gemini re-run complete for {file_name}. New Plate: {new_plate_text if new_plate_text else 'NOT FOUND'}")
                                st.rerun()

                        # Button 3: Save to MongoDB
                        if col_actions_3.button("Save to MongoDB", 
                                                key=f"save_db_{i}", 
                                                disabled=record['saved'],
                                                help="Save the current license plate and vehicle specifications to the MongoDB database."):
                            with st.spinner(f'Saving **{file_name}** to MongoDB...'):
                                success, result_id = save_data_to_mongodb(
                                    file_name=file_name,
                                    file_bytes=file_bytes,
                                    mime_type=mime_type,
                                    plate_text=record['plate_text'],
                                    plate_status=record['plate_status'],
                                    specs=record['specs']
                                )
                                
                                if success:
                                    st.session_state['analysis_results'][i]['saved'] = True
                                    alert_message = f"Data for {file_name} successfully saved. DB ID: {result_id}"
                                    show_alert(alert_message)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to save **{file_name}**: {result_id}")


                st.markdown("---") # Separator between records
            
            
            # --- SAVE ALL BUTTON SECTION (at the bottom) ---
            unsaved_records = [r for r in st.session_state['analysis_results'] if not r['saved']]
            total_unsaved = len(unsaved_records)
            
            if total_unsaved > 0:
                st.markdown("---")
                
                # Button to save all remaining records
                if st.button(f"💾 Save All {total_unsaved} Unsaved Records to MongoDB", 
                             type="primary", 
                             help="Save all records that have not yet been marked as saved to the MongoDB database in a single action."):
                    
                    saved_count = 0
                    failed_count = 0
                    
                    with st.spinner(f'Starting bulk save of {total_unsaved} records...'):
                        
                        # Iterate through the session state and save unsaved records
                        for i, record in enumerate(st.session_state['analysis_results']):
                            if not record['saved']:
                                success, result_id = save_data_to_mongodb(
                                    file_name=record['file_name'],
                                    file_bytes=record['file_bytes'],
                                    mime_type=record['mime_type'],
                                    plate_text=record['plate_text'],
                                    plate_status=record['plate_status'],
                                    specs=record['specs']
                                )
                                
                                if success:
                                    st.session_state['analysis_results'][i]['saved'] = True
                                    saved_count += 1
                                else:
                                    failed_count += 1
                                    st.error(f"❌ Failed to save {record['file_name']}: {result_id}")
                                    
                    st.toast(f"Bulk Save Complete: {saved_count} saved, {failed_count} failed.")
                    if saved_count > 0:
                        show_alert(f"Bulk save completed! Successfully saved {saved_count} record(s).")
                    
                    st.rerun() # Rerun to update button status and UI
                
            # Final status check (runs after save all is processed or if everything was saved)
            if all(record['saved'] for record in st.session_state['analysis_results']):
                st.balloons()
                st.success("All analyzed records have been successfully saved to MongoDB.")
                
        else:
            st.info(f"Click the Analyze Vehicle(s) button above to begin initial analysis for the {len(uploaded_files)} uploaded image(s).")
else:
    st.info("Please upload one or more images to begin.")

# --- Footer Section (Unchanged) ---
st.markdown("---")

footer_html = """
<div style="text-align: center; color: grey; font-size: 0.8em; padding: 10px 0;">
    &copy; 2025 Butter Idea Sdn Bhd. All rights reserved. 
    <br> 
    Powered by <a href="https://www.bi2u.com/" target="_blank" style="color: grey; text-decoration: underline;">bi2u</a>.
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)