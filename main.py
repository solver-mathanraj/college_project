import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import sqlite3

# --- Database Functions ---
def init_db():
    conn = sqlite3.connect('database/users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def signup(username, password):
    conn = sqlite3.connect('database/users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login(username, password):
    conn = sqlite3.connect('database/users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

# --- Page Setup ---
st.set_page_config(page_title="Helmet Detection", layout="centered")

# --- Initialize ---
init_db()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Styling ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ffccd5 0%, #a1c4fd 100%);
    color: #333;
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #d81b60;
    text-align: center;
    font-size: 2.5rem;
}
.stSidebar {
    background: linear-gradient(180deg, #f06292 0%, #ffccbc 100%);
}
.stSidebar h2 {
    color: white;
    text-align: center;
}
.stRadio > div {
    display: flex;
    justify-content: center;
    gap: 1rem;
}
.stRadio > label {
    color: white;
    font-weight: bold;
}
.stTextInput input {
    # background: #fff;
    border-radius: 5px;
}
.stButton > button {
    background: linear-gradient(90deg, #d81b60 0%, #f06292 100%);
    color: white;
    border-radius: 5px;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #c2185b 0%, #ec407a 100%);
}
.upload-box {
    # background: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    max-width: 500px;
    margin: 20px auto;
    text-align:center;
}
.image-container img {
    max-width: 100%;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.prediction {
    background: linear-gradient(90deg, #c8e6c9 0%, #a5d6a7 100%);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("User")
    if st.session_state.logged_in:
        st.success(f"Hi, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    else:
        mode = st.radio("", ["Login", "Signup"])
        if mode == "Login":
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error("Wrong credentials")
        else:
            with st.form("signup"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                if st.form_submit_button("Signup"):
                    if signup(new_username, new_password):
                        st.session_state.logged_in = True
                        st.session_state.username = new_username
                        st.success("Signed up!")
                        st.rerun()
                    else:
                        st.error("Username taken")

# --- Model and Prediction ---
model = YOLO('model/helmet_yolov8.pt')
confidence_threshold = 0.5
class_names = ['helmet', 'no_helmet']  # Updated based on previous correction

def detect_helmets(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image")

    # Perform inference
    results = model(image, conf=confidence_threshold)

    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{class_names[cls]} {conf:.2f}"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for helmet, red for no_helmet
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detections.append(label)

    # Convert image for Streamlit display (BGR to RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, detections

# --- Main ---
st.title("Helmet Detection")
if st.session_state.logged_in:
    st.markdown('<div class="upload-box">Upload an image to detect helmets.</div>', unsafe_allow_html=True)
    file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name
        try:
            
            # Perform helmet detection
            processed_image, detections = detect_helmets(path)
            
            # Display the processed image with detections
            st.image(processed_image, caption="Detected Helmets", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        finally:
            os.remove(path)
else:
    st.info("Please log in or sign up to use the app.")