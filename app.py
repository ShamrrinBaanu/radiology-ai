import streamlit as st
import os
import database
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


database.create_tables()
if "selected_scan" not in st.session_state:
    st.session_state.selected_scan = None
    
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

os.makedirs("scans", exist_ok=True)
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    model.load_state_dict(torch.load(
        r"C:\Users\Shamrrin S\Desktop\radiology-image-classification\models\brain_ct_model.pth",
        map_location="cpu"
    ))

    model.eval()
    return model

model = load_model()
# ---------------- HEATMAP FUNCTION ----------------

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def generate_heatmap(image_path):

    # Load image exactly like training
    pil_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(pil_image).unsqueeze(0)

    # Convert same image for visualization
    rgb_img = np.array(pil_image.resize((224,224))).astype(np.float32) / 255

    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization

def hemorrhage_area(image_path):

    pil_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(pil_image).unsqueeze(0)

    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # threshold to detect strong activation (possible hemorrhage)
    mask = grayscale_cam > 0.6

    area = np.sum(mask)

    return area

def hemorrhage_progression(scan1, scan2):

    area1 = hemorrhage_area(scan1)
    area2 = hemorrhage_area(scan2)

    if area1 == 0:
        return 0, "No previous hemorrhage detected"

    change = ((area2 - area1) / area1) * 100

    if change > 0:
        status = f"🔺 Bleeding Increased by {change:.2f}%"
    elif change < 0:
        status = f"🔻 Bleeding Reduced by {abs(change):.2f}%"
    else:
        status = "➖ No significant change"

    return change, status


# ---------------- TRIAGE FUNCTION ----------------

def triage_level(prediction, confidence):

    if prediction == "Hemorrhage":

        if confidence > 95:
            return "CRITICAL", "🔴"

        elif confidence > 85:
            return "HIGH", "🟠"

        else:
            return "MODERATE", "🟡"

    else:
        return "NORMAL", "🟢"

# ---------------- SIDEBAR ----------------

def sidebar():

    st.sidebar.title("🏥 Radiology AI")

    patients = database.get_patients()

    st.sidebar.metric("Total Patients",len(patients))

    st.sidebar.markdown("---")

    search = st.sidebar.text_input("🔎 Search Patient")

    for p in patients:

        if search.lower() in p[1].lower():

            if st.sidebar.button(f"👤 {p[1]}",key=f"p{p[0]}"):

                st.session_state.selected_patient=p
                st.session_state.page="patient"
                st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("📊 Dashboard"):
        st.session_state.page="dashboard"
        st.rerun()

    if st.sidebar.button("📂 Reports"):
        st.session_state.page="reports"
        st.rerun()

    if st.sidebar.button("⚙ Settings"):
        st.session_state.page="settings"
        st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in=False
        st.session_state.page="login"
        st.rerun()


# ---------------- LOGIN ----------------

def login_page():

    st.title("🏥 Radiology AI Login")

    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if st.button("Login"):

        doctor=database.verify_doctor(username,password)

        if doctor:
            st.session_state.logged_in=True
            st.session_state.page="dashboard"
            st.rerun()
        else:
            st.error("Invalid login")

    if st.button("Forgot Password"):
        st.session_state.page="forgot_password"
        st.rerun()


# ---------------- FORGOT PASSWORD ----------------

def forgot_password():

    st.title("Reset Password")

    username=st.text_input("Username")
    new_pass=st.text_input("New Password",type="password")

    if st.button("Reset Password"):

        database.reset_password(username,new_pass)

        st.success("Password Updated")

        st.session_state.page="login"
        st.rerun()


# ---------------- DASHBOARD ----------------

def dashboard():

    st.title("Radiology AI Dashboard")

    patients=database.get_patients()

    st.subheader("Patients")

    for p in patients:

        scans=database.get_scans(p[0])

        status="🟢 Normal"

        if scans:
            if scans[0][1]=="Hemorrhage":
                status="🔴 Critical"

        col1,col2=st.columns([4,1])

        with col1:
            st.write(f"{p[1]} | Age:{p[2]} | Gender:{p[3]} | {status}")

        with col2:
            if st.button("Open",key=p[0]):
                st.session_state.selected_patient=p
                st.session_state.page="patient"
                st.rerun()

    st.markdown("---")

    st.subheader("Add Patient")

    name=st.text_input("Patient Name")
    age=st.number_input("Age",0,120)
    gender=st.selectbox("Gender",["Male","Female","Other"])

    if st.button("Add Patient"):
        database.add_patient(name,age,gender)
        st.success("Patient Added")
        st.rerun()


def predict_scan(image_path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # open image correctly
    image = Image.open(image_path).convert("RGB")

    # apply transform
    image = transform(image)

    # add batch dimension
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs,1)

    classes = ["Hemorrhage","Normal"]

    prediction = classes[predicted.item()]
    confidence = confidence.item()*100

    return prediction, confidence
# ---------------- PATIENT PAGE ----------------

def patient_page():

    patient = st.session_state.selected_patient

    st.title(patient[1])

    st.markdown("---")

    st.subheader("Upload New Scan")

    file = st.file_uploader("Upload Scan")

    if file:

        path = f"scans/{file.name}"

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        st.image(path)

        col1, col2 = st.columns(2)

        with col1:
            run = st.button("Run AI Prediction")

        with col2:
            compare = st.button("Compare Last Two Scans")

        if run:

            prediction, confidence = predict_scan(path)

            database.add_scan(patient[0], path, prediction, confidence)

            level, icon = triage_level(prediction, confidence)

            st.markdown("## 🧠 AI Analysis")

            heatmap = generate_heatmap(path)

            st.image(heatmap, caption="AI Attention Map")

            st.markdown("### 📊 Confidence Level")

            st.progress(confidence / 100)

            st.write(f"{confidence:.2f}% confidence")

            st.markdown("### 🚑 Emergency Triage Level")

            st.markdown(f"## {icon} {level}")

        if compare:
            st.session_state.page = "compare"
            st.rerun()

    # ---------------- SCAN HISTORY ----------------

    st.markdown("---")

    with st.expander("📂 Scan History", expanded=False):

        scans = database.get_scans(patient[0])

        if not scans:
            st.info("No previous scans available")

        for i, s in enumerate(scans):

            with st.container():

                col1, col2, col3 = st.columns([1,3,1])

                with col1:
                    if os.path.exists(s[0]):
                        st.image(s[0], width=100)

                with col2:
                    st.markdown(f"**🧠 Scan {i+1}**")
                    st.write("Prediction:", s[1])
                    st.write("Confidence:", f"{s[2]:.2f}%")
                    st.write("Date:", s[3])

                with col3:
                    if st.button("👁 View", key=f"view{i}"):

                        st.session_state.selected_scan = s
                        st.session_state.page = "view_scan"
                        st.rerun()

                st.markdown("---")

# ---------------- COMPARE SCANS ----------------

def compare_scans():

    patient = st.session_state.selected_patient

    st.title("📊 Scan Comparison")

    scans = database.get_last_two_scans(patient[0])

    if len(scans) < 2:
        st.warning("Need 2 scans to compare")
        return

    latest = scans[0][0]
    previous = scans[1][0]

    col1, col2 = st.columns(2)

    with col1:
        st.image(latest)
        st.write("Latest Scan")

    with col2:
        st.image(previous)
        st.write("Previous Scan")

    st.markdown("---")

    change, status = hemorrhage_progression(previous, latest)

    st.markdown("## 🧠 AI Disease Progression Analysis")

    if change > 10:
        st.error(status)

    elif change > 0:
        st.warning(status)

    elif change < 0:
        st.success(status)

    else:
        st.info(status)

def view_scan_page():

    scan = st.session_state.selected_scan

    st.title("🧠 Scan Details")

    if scan:

        col1, col2 = st.columns(2)

        with col1:
            st.image(scan[0], caption="CT Scan", use_container_width=True)

        with col2:
            st.subheader("AI Results")
            st.write("Prediction:", scan[1])
            st.write("Confidence:", f"{scan[2]:.2f}%")
            st.write("Date:", scan[3])

    if st.button("⬅ Back to Patient"):
        st.session_state.page = "patient"
        st.rerun()
        
        
# ---------------- REPORTS ----------------

def reports_page():

    st.title("All Scan Reports")

    patients=database.get_patients()

    for p in patients:

        scans=database.get_scans(p[0])

        if scans:

            st.subheader(p[1])

            for s in scans:

                col1,col2=st.columns([1,3])

                with col1:
                    if os.path.exists(s[0]):
                        st.image(s[0],width=120)

                with col2:
                    st.write("Prediction:",s[1])
                    st.write("Confidence:",s[2])
                    st.write("Date:",s[3])


# ---------------- SETTINGS ----------------

def settings_page():

    st.title("Settings")

    username=st.text_input("Doctor Name")
    hospital=st.text_input("Hospital")
    email=st.text_input("Email")

    if st.button("Save"):
        st.success("Settings Saved")


# ---------------- ROUTER ----------------

if st.session_state.page=="login":
    login_page()

elif st.session_state.page=="forgot_password":
    forgot_password()

else:

    sidebar()

    if st.session_state.page=="dashboard":
        dashboard()

    elif st.session_state.page=="patient":
        patient_page()

    elif st.session_state.page=="compare":
        compare_scans()
        
    elif st.session_state.page=="view_scan":
        view_scan_page()
    
    elif st.session_state.page=="reports":
        reports_page()

    elif st.session_state.page=="settings":
        settings_page()
        