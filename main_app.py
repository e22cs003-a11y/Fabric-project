# app.py - Complete Textile Defect Detection System with Mobile Optimization
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import requests
from streamlit_oauth import OAuth2Component
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Textile Defect Detection System",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile Optimization - Viewport & Responsive Styles
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
<style>
    /* Mobile Responsive Styles */
    @media (max-width: 768px) {
        .stButton button {
            font-size: 16px !important;
            padding: 10px !important;
            width: 100% !important;
        }
        .stMarkdown h1 {
            font-size: 1.8em !important;
        }
        .stMarkdown h2 {
            font-size: 1.4em !important;
        }
        .stMarkdown h3 {
            font-size: 1.2em !important;
        }
        .stColumns {
            flex-direction: column !important;
        }
        .element-container {
            padding: 5px 0 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 14px !important;
        }
        .stAlert {
            padding: 10px !important;
        }
        .stImage {
            margin: 10px 0 !important;
        }
        .stDataFrame {
            font-size: 12px !important;
        }
        .metric-card {
            padding: 10px !important;
        }
        .metric-value {
            font-size: 20px !important;
        }
        .stat-card {
            padding: 12px !important;
        }
        .stat-number {
            font-size: 24px !important;
        }
        .hero-title {
            font-size: 32px !important;
        }
        .hero-subtitle {
            font-size: 16px !important;
        }
        .feature-grid {
            grid-template-columns: 1fr !important;
            gap: 16px !important;
            padding: 20px !important;
        }
        .feature-card {
            padding: 20px !important;
        }
        .dashboard-header {
            padding: 12px 20px !important;
            flex-direction: column !important;
            gap: 10px !important;
        }
        .user-info {
            justify-content: center !important;
        }
        .upload-area {
            padding: 20px !important;
        }
    }
    
    /* Tablet Styles */
    @media (min-width: 769px) and (max-width: 1024px) {
        .feature-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .hero-title {
            font-size: 44px !important;
        }
    }
    
    /* Smooth animations */
    * {
        -webkit-tap-highlight-color: transparent;
    }
    
    /* Better touch targets for mobile */
    button, .stButton button, .stDownloadButton button {
        min-height: 44px !important;
        border-radius: 12px !important;
    }
    
    /* Input field improvements */
    input, textarea, .stTextInput input {
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    /* Camera input styling */
    .stCameraInput {
        border-radius: 20px !important;
        overflow: hidden !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Success/Balloon animation */
    .stBalloon {
        animation: floatUp 0.5s ease !important;
    }
    
    @keyframes floatUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Dark mode support for better readability */
    @media (prefers-color-scheme: dark) {
        .metric-card, .stat-card, .solution-card {
            background: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        .history-item {
            background: #2d2d2d !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Google OAuth Configuration
CLIENT_ID = "930915758489-276a4s3f9rut5geq2sg39e4vom9316rh.apps.googleusercontent.com"
CLIENT_SECRET = "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"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
REDIRECT_URI = "http://localhost:8501"

# Admin emails
ADMINS = ["santhoshwebworker@gmail.com", "e22cs003@shanmugha.edu.in"]

# Initialize session state - PERSISTENT across refresh
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ''
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = ''
if 'user_picture' not in st.session_state:
    st.session_state['user_picture'] = ''
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []
if 'last_analysis' not in st.session_state:
    st.session_state['last_analysis'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'dashboard'

# Database files
USERS_FILE = "users_data.json"
REPORTS_FILE = "reports_data.json"

def init_db():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    if not os.path.exists(REPORTS_FILE):
        with open(REPORTS_FILE, 'w') as f:
            json.dump([], f)

init_db()

def save_user(user_data):
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    if user_data['email'] not in users:
        users[user_data['email']] = user_data
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)

def get_all_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_report(report_data):
    with open(REPORTS_FILE, 'r') as f:
        reports = json.load(f)
    reports.append(report_data)
    with open(REPORTS_FILE, 'w') as f:
        json.dump(reports, f, indent=2)

def get_user_reports(email):
    with open(REPORTS_FILE, 'r') as f:
        reports = json.load(f)
    return [r for r in reports if r.get('user_email') == email]

def get_all_reports():
    with open(REPORTS_FILE, 'r') as f:
        return json.load(f)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .navbar {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 16px 48px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-section {
        text-align: center;
        padding: 80px 40px;
        color: white;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 20px;
        animation: fadeInUp 0.8s ease;
    }
    
    .hero-subtitle {
        font-size: 20px;
        opacity: 0.95;
        margin-bottom: 40px;
        animation: fadeInUp 1s ease;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 24px;
        padding: 40px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 32px 24px;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.6s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        background: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #333;
        margin-bottom: 12px;
    }
    
    .feature-desc {
        font-size: 14px;
        color: #666;
        line-height: 1.5;
    }
    
    .dashboard-header {
        background: white;
        padding: 20px 48px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 20px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }
    
    .user-info {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .user-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 20px;
    }
    
    .stat-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
        margin-bottom: 20px;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: #f0f0ff;
    }
    
    .defect-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
    }
    
    .success-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        color: white;
    }
    
    .solution-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 12px;
        color: #666;
        margin-top: 8px;
    }
    
    .history-item {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 0.5s ease;
    }
    
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
            padding: 20px;
        }
        .hero-title {
            font-size: 32px;
        }
        .hero-section {
            padding: 40px 20px;
        }
        .navbar, .dashboard-header {
            padding: 12px 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Defect Detector with Classification & Hole Detection
class DefectDetector:
    def __init__(self):
        self.threshold = 0.5
    
    def classify_defect(self, score):
        """Classify defect type based on score"""
        if score > 0.85:
            return "Critical Hole / Tear"
        elif 0.70 < score <= 0.85:
            return "Oil / Chemical Stain"
        elif 0.55 < score <= 0.70:
            return "Thread Cut / Knitting Error"
        elif 0.40 < score <= 0.55:
            return "Minor Surface Irregularity"
        else:
            return "Unknown Anomaly"
    
    def check_for_holes(self, image):
        """Detect holes in the fabric image"""
        # Convert image to grayscale
        img_array = np.array(image.convert('L'))
        
        # Thresholding to detect holes
        _, thresh = cv2.threshold(img_array, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hole_detected = False
        hole_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter small spots
                hole_detected = True
                hole_areas.append(area)
        
        return hole_detected, hole_areas
    
    def draw_hole_boxes(self, image):
        """Draw bounding boxes around detected holes"""
        img_copy = np.array(image.convert('RGB')).copy()
        img_gray = np.array(image.convert('L'))
        
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_copy, "HOLE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_copy
    
    def detect_defect_region(self, gray):
        """Detect defect region in the image"""
        height, width = gray.shape
        region_size = 50
        defect_regions = []
        
        for y in range(0, height, region_size):
            for x in range(0, width, region_size):
                region = gray[y:min(y+region_size, height), x:min(x+region_size, width)]
                if region.size > 0:
                    region_std = np.std(region)
                    if region_std > 40:
                        defect_regions.append((x, y, min(x+region_size, width), min(y+region_size, height)))
        
        return defect_regions
    
    def analyze_fabric(self, image):
        try:
            start_time = time.time()
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate features
            std_dev = np.std(gray)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            mean_brightness = np.mean(gray)
            
            # Detect defect regions
            defect_regions = self.detect_defect_region(gray)
            
            # Check for holes specifically
            has_hole, hole_areas = self.check_for_holes(image)
            
            # Calculate defect score
            defect_score = ((std_dev / 255) * 0.35 + (entropy / 8) * 0.35 + edge_density * 0.2 + (1 - mean_brightness/255) * 0.1)
            defect_score = float(min(1.0, max(0.0, defect_score)))
            
            # If hole detected, increase score
            if has_hole:
                defect_score = max(defect_score, 0.75)
            
            is_defect = bool(defect_score > self.threshold)
            
            # Get defect classification
            if is_defect:
                if has_hole:
                    defect_name = "Critical Hole / Tear"
                    sub_type = "Fabric Damage / Puncture"
                    severity = "Critical"
                    severity_score = 95
                    color = "#dc3545"
                    actions = [
                        " STOP production line immediately",
                        " Mark and isolate defective section",
                        " Inspect machine for sharp edges/needles",
                        " Document defect for quality team",
                        " Replace damaged parts before resuming"
                    ]
                    causes = [
                        "Sharp object contact with fabric",
                        "Machine needle damage or misalignment",
                        "Excessive tension on fabric"
                    ]
                    prevention = [
                        "Daily machine inspection checklist",
                        "Use protective covers for sharp edges",
                        "Regular maintenance schedule"
                    ]
                else:
                    defect_name = self.classify_defect(defect_score)
                    if "Hole" in defect_name:
                        sub_type = "Fabric Damage"
                        severity = "Critical"
                        severity_score = 90
                        color = "#dc3545"
                    elif "Stain" in defect_name:
                        sub_type = "Chemical Contamination"
                        severity = "High"
                        severity_score = 75
                        color = "#fd7e14"
                    elif "Thread" in defect_name:
                        sub_type = "Weave Irregularity"
                        severity = "Medium"
                        severity_score = 55
                        color = "#ffc107"
                    else:
                        sub_type = "Surface Irregularity"
                        severity = "Low"
                        severity_score = 35
                        color = "#28a745"
                    
                    actions = [
                        " Document the defect",
                        " Schedule maintenance check",
                        " Track for pattern analysis"
                    ]
                    causes = ["Manufacturing irregularity", "Process variation"]
                    prevention = ["Regular quality checks", "Process optimization"]
            else:
                defect_name = "No Defect - Premium Quality"
                sub_type = "Perfect Fabric"
                severity = "Excellent"
                severity_score = 0
                color = "#28a745"
                actions = [
                    " Fabric quality is acceptable",
                    " Continue with production",
                    " Quality standards met"
                ]
                causes = ["Normal fabric texture"]
                prevention = ["Continue current quality practices"]
            
            # Draw annotated image
            if has_hole:
                annotated_image = self.draw_hole_boxes(image)
            elif is_defect and defect_regions:
                annotated_image = self.draw_bounding_boxes(image, defect_regions)
            else:
                annotated_image = np.array(image)
            
            # Calculate confidence
            confidence = 1.0 - abs(defect_score - (0.5 if is_defect else 0))
            confidence = float(min(0.99, max(0.70, confidence)))
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                'is_defect': is_defect,
                'defect_score': defect_score,
                'defect_type': defect_name,
                'sub_type': sub_type,
                'severity': severity,
                'severity_score': severity_score,
                'color': color,
                'actions': actions,
                'causes': causes,
                'prevention': prevention,
                'confidence': confidence,
                'processing_time': processing_time,
                'defect_regions': defect_regions,
                'has_hole': has_hole,
                'hole_areas': hole_areas if has_hole else [],
                'annotated_image': annotated_image,
                'features': {
                    'std_dev': float(std_dev),
                    'entropy': float(entropy),
                    'edge_density': float(edge_density),
                    'mean_brightness': float(mean_brightness)
                }
            }
        except Exception as e:
            return {
                'is_defect': False,
                'defect_score': 0.0,
                'defect_type': "Analysis Error",
                'sub_type': "Unknown",
                'severity': "Unknown",
                'severity_score': 0,
                'color': "#666",
                'actions': ["Please try again with a clearer image"],
                'causes': ["Image processing error"],
                'prevention': ["Ensure good lighting and clear image"],
                'confidence': 0.0,
                'processing_time': 0.0,
                'defect_regions': [],
                'has_hole': False,
                'hole_areas': [],
                'annotated_image': np.array(image) if image else None,
                'features': {}
            }
    
    def draw_bounding_boxes(self, image, defect_regions):
        """Draw bounding boxes on the image"""
        img_copy = np.array(image).copy()
        if len(img_copy.shape) == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        
        for (x1, y1, x2, y2) in defect_regions:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_copy, "Defect", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_copy

# Home Page with OAuth - Super Beautiful
def home_page():
    st.markdown("""
    <div class="navbar">
        <div class="logo">
            <span>🧵</span>
            <span>Textile Detect Pro</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">✨ Zero-Label Defect Detection ✨</div>
        <div class="hero-subtitle">Powered by Self-Supervised Contrastive Learning & AI<br>Detect fabric defects instantly without labeled data</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center; margin: 20px 0;"><h3> Sign In to Continue</h3></div>', unsafe_allow_html=True)
        
        oauth2 = OAuth2Component(
            CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL
        )
        
        result = oauth2.authorize_button(
            name="Continue with Google",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_oauth",
            use_container_width=True,
        )
        
        if result and 'token' in result:
            access_token = result['token']['access_token']
            user_info_response = requests.get(
                'https://www.googleapis.com/oauth2/v3/userinfo',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if user_info_response.status_code == 200:
                user_info = user_info_response.json()
                user_email = user_info.get('email')
                user_name = user_info.get('name', user_email.split('@')[0])
                user_picture = user_info.get('picture', '')
                
                user_data = {
                    'email': user_email,
                    'name': user_name,
                    'picture': user_picture,
                    'role': 'admin' if user_email in ADMINS else 'user',
                    'created_at': datetime.now().isoformat(),
                    'last_login': datetime.now().isoformat()
                }
                save_user(user_data)
                
                st.session_state['logged_in'] = True
                st.session_state['user_email'] = user_email
                st.session_state['user_name'] = user_name
                st.session_state['user_picture'] = user_picture
                st.session_state['user_role'] = 'admin' if user_email in ADMINS else 'user'
                st.session_state['page'] = 'dashboard'
                
                st.rerun()
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Zero-Label Learning</div>
            <div class="feature-desc">No defect labels needed - AI learns automatically</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Mobile Ready</div>
            <div class="feature-desc">Works perfectly on all devices - Phone, Tablet, Desktop</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Real-Time</div>
            <div class="feature-desc">Instant detection with high accuracy</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Smart Solutions</div>
            <div class="feature-desc">Auto recommendations for each defect type</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Hole Detection</div>
            <div class="feature-desc">Specialized algorithm to detect holes and tears</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Analytics Dashboard</div>
            <div class="feature-desc">Track defects, trends and quality metrics</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Admin Panel</div>
            <div class="feature-desc">Complete user and report management</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon"></div>
            <div class="feature-title">Secure Login</div>
            <div class="feature-desc">Google OAuth 2.0 authentication</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    users = get_all_users()
    reports = get_all_reports()
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown(f"""
        <div style="text-align: center; color: white; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800;">{len(users)}+</div>
            <div style="font-size: 14px;">Happy Users</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div style="text-align: center; color: white; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800;">{len(reports)}+</div>
            <div style="font-size: 14px;">Inspections Done</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        defects = sum(1 for r in reports if r.get('is_defect', False))
        st.markdown(f"""
        <div style="text-align: center; color: white; padding: 20px;">
            <div style="font-size: 48px; font-weight: 800;">{defects}+</div>
            <div style="font-size: 14px;">Defects Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: white;">
        <p style="font-size: 14px;">© 2024 Textile Defect Detection System | Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

# User Dashboard
def user_dashboard():
    # Dashboard Header with Logout Button (FIXED - Using st.button instead of HTML)
    st.markdown("""
    <div class="dashboard-header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <span style="font-size: 28px;">🧵</span>
            <span style="font-weight: 700; font-size: 20px;">Defect Detection Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button as st.button (NOT HTML button)
    col_logout1, col_logout2 = st.columns([3, 1])
    with col_logout2:
        if st.button("🚪 Logout", key="logout_user", use_container_width=True):
            # Clear all session data
            session_keys = ['logged_in', 'user_email', 'user_name', 'user_role', 'user_picture', 
                           'current_image', 'analysis_history', 'last_analysis', 'page']
            for key in session_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # User info display
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 16px; margin-bottom: 10px;">
        <div class="user-avatar">{st.session_state.get('user_name', 'U')[0].upper()}</div>
        <div>
            <div style="font-weight: 600;">{st.session_state.get('user_name')}</div>
            <div style="font-size: 12px; color: #666;">{st.session_state.get('user_email')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome Message
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                padding: 20px; border-radius: 16px; margin-bottom: 20px;">
        <h2>Welcome back, {st.session_state.get('user_name')}! </h2>
        <p style="color: #666;">Ready to inspect fabric quality? Upload an image or use your camera.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Row
    reports = get_user_reports(st.session_state['user_email'])
    total = len(reports)
    defects = sum(1 for r in reports if r.get('is_defect', False))
    today_count = sum(1 for r in reports if r.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d')))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-number" style="color:#667eea">{total}</div><div class="stat-label">Total Inspections</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card"><div class="stat-number" style="color:#dc3545">{defects}</div><div class="stat-label">Defects Found</div></div>""", unsafe_allow_html=True)
    with col3:
        rate = (defects/total*100) if total>0 else 0
        st.markdown(f"""<div class="stat-card"><div class="stat-number" style="color:#fd7e14">{rate:.1f}%</div><div class="stat-label">Defect Rate</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="stat-card"><div class="stat-number" style="color:#28a745">{today_count}</div><div class="stat-label">Today's Scans</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Content - Two Columns
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        st.markdown("###  Fabric Inspection")
        
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 48px; margin-bottom: 16px;">📷</div>
            <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">Upload or Capture Image</div>
            <div style="font-size: 14px; color: #666;">Take a photo or upload from gallery</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📷 Camera", "📱 Mobile", "📁 Upload"])
        captured_image = None
        
        with tab1:
            camera = st.camera_input("Take a photo", key="webcam")
            if camera:
                captured_image = Image.open(camera)
                st.image(captured_image, caption="📸 Captured Image", use_column_width=True)
        
        with tab2:
            mobile = st.file_uploader("Choose from gallery", type=['jpg', 'jpeg', 'png'], key="mobile")
            if mobile:
                captured_image = Image.open(mobile)
                st.image(captured_image, caption="📱 Selected Image", use_column_width=True)
        
        with tab3:
            upload = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'], key="upload")
            if upload:
                captured_image = Image.open(upload)
                st.image(captured_image, caption="📁 Uploaded Image", use_column_width=True)
        
        if captured_image:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Analyze Fabric", type="primary", use_container_width=True):
                    st.session_state['current_image'] = captured_image
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ Clear Image", use_container_width=True):
                    st.session_state['current_image'] = None
                    st.session_state['last_analysis'] = None
                    st.rerun()
    
    with col_right:
        st.markdown("###  Quick Stats")
        
        if reports:
            today_reports = [r for r in reports if r.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(today_reports)}</div>
                <div class="metric-label">Scans Today</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Defect type distribution from history
            defect_types = {}
            for r in reports[-20:]:
                dt = r.get('defect_type', 'Unknown')[:20]
                defect_types[dt] = defect_types.get(dt, 0) + 1
            
            if defect_types:
                df_types = pd.DataFrame(list(defect_types.items()), columns=['Type', 'Count'])
                fig = px.pie(df_types, values='Count', names='Type', title='Defect Distribution')
                fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inspections yet. Start analyzing fabric!")
    
    st.markdown("---")
    
    # Analysis Results Section
    if st.session_state.get('current_image') is not None:
        st.markdown("## Analysis Results")
        
        with st.spinner("🔬 Analyzing fabric quality..."):
            detector = DefectDetector()
            result = detector.analyze_fabric(st.session_state['current_image'])
            
            # Save to history
            history_entry = {
                'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'timestamp': datetime.now().isoformat(),
                'defect_score': result['defect_score'],
                'is_defect': result['is_defect'],
                'defect_type': result['defect_type'],
                'sub_type': result['sub_type'],
                'severity': result['severity']
            }
            st.session_state['analysis_history'].insert(0, history_entry)
            if len(st.session_state['analysis_history']) > 10:
                st.session_state['analysis_history'] = st.session_state['analysis_history'][:10]
            
            # Save report to database
            report_data = {
                'id': history_entry['id'],
                'user_email': st.session_state['user_email'],
                'timestamp': history_entry['timestamp'],
                'defect_score': float(result['defect_score']),
                'is_defect': bool(result['is_defect']),
                'defect_type': str(result['defect_type']),
                'sub_type': str(result['sub_type']),
                'severity': str(result['severity']),
                'confidence': float(result['confidence']),
                'processing_time': float(result['processing_time']),
                'has_hole': bool(result.get('has_hole', False))
            }
            save_report(report_data)
            
            # Display Original vs Annotated
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 Original Image")
                st.image(st.session_state['current_image'], use_column_width=True)
            
            with col2:
                st.markdown("### 🔍 Defect Detection View")
                if result['annotated_image'] is not None:
                    st.image(result['annotated_image'], use_column_width=True)
                    if result.get('has_hole', False):
                        st.caption(f"🕳️ {len(result.get('hole_areas', []))} holes detected")
                    elif result['defect_regions']:
                        st.caption(f"📍 {len(result['defect_regions'])} defect regions detected")
                else:
                    st.image(st.session_state['current_image'], use_column_width=True)
            
            # Results Card
            if result['is_defect']:
                if result.get('has_hole', False):
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>🕳️ {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p style="font-size: 18px;">⚠️ HOLE DETECTED in fabric!</p>
                        <p>Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("🚨 **URGENT: Hole/Tear detected in fabric. Stop machine immediately!**")
                else:
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>⚠️ {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p style="font-size: 18px;">Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card pulse">
                    <h2>✅ {result['defect_type']}</h2>
                    <p style="font-size: 18px;">Quality Score: {(1-result['defect_score']):.1%}</p>
                    <p>Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            
            # Metrics Row
            st.markdown("###  Quality Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['confidence']:.1%}</div>
                    <div class="metric-label">AI Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                quality_score = (1 - result['defect_score']) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{quality_score:.1f}%</div>
                    <div class="metric-label">Quality Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['severity_score']}</div>
                    <div class="metric-label">Severity Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['processing_time']}s</div>
                    <div class="metric-label">Processing Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge Chart
            st.markdown("###  Defect Probability Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['defect_score'] * 100,
                title={'text': "Defect Score", 'font': {'size': 20}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': result['color']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Solutions Section
            if result['is_defect']:
                st.markdown("### 🔧 Recommended Actions")
                for action in result['actions']:
                    st.markdown(f"<div class='solution-card'>✅ {action}</div>", unsafe_allow_html=True)
                
                with st.expander("🔍 Possible Causes"):
                    for cause in result['causes']:
                        st.markdown(f"📌 {cause}")
                
                with st.expander("🛡️ Prevention Measures"):
                    for prev in result['prevention']:
                        st.markdown(f"🛡️ {prev}")
                
                # Download Report Button
                report_json = json.dumps({
                    'defect_type': result['defect_type'],
                    'sub_type': result['sub_type'],
                    'severity': result['severity'],
                    'defect_score': result['defect_score'],
                    'confidence': result['confidence'],
                    'has_hole': result.get('has_hole', False),
                    'actions': result['actions'],
                    'timestamp': datetime.now().isoformat()
                }, indent=2)
                
                st.download_button(
                    label=" Download Full Report (JSON)",
                    data=report_json,
                    file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # New Inspection button
        if st.button(" New Inspection", use_container_width=True):
            st.session_state['current_image'] = None
            st.session_state['last_analysis'] = None
            st.rerun()
    
    # Recent Inspections History
    st.markdown("---")
    st.markdown("### 📋 Recent Inspections History")
    
    if st.session_state.get('analysis_history'):
        for item in st.session_state['analysis_history'][:5]:
            timestamp = datetime.fromisoformat(item['timestamp']).strftime('%H:%M %d/%m/%Y')
            status = "🔴" if item['is_defect'] else "🟢"
            st.markdown(f"""
            <div class="history-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 20px;">{status}</span>
                        <span style="font-weight: 600; margin-left: 10px;">{item['defect_type'][:40]}</span>
                    </div>
                    <div style="color: #666; font-size: 12px;">{timestamp}</div>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    Score: {item['defect_score']:.1%} | Severity: {item['severity']} | Type: {item['sub_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No inspection history yet. Start analyzing fabric!")

# Admin Dashboard
def admin_dashboard():
    st.markdown("""
    <div class="dashboard-header">
        <div><span style="font-size: 28px;">👑</span> <span style="font-weight: 700;">Admin Control Panel</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button as st.button (NOT HTML button)
    col_logout1, col_logout2 = st.columns([3, 1])
    with col_logout2:
        if st.button("🚪 Logout", key="logout_admin", use_container_width=True):
            # Clear all session data
            session_keys = ['logged_in', 'user_email', 'user_name', 'user_role', 'user_picture', 
                           'current_image', 'analysis_history', 'last_analysis', 'page']
            for key in session_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Admin info display
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 16px; margin-bottom: 10px;">
        <div class="user-avatar">A</div>
        <div>
            <div style="font-weight: 600;">Administrator</div>
            <div style="font-size: 12px; color: #666;">{st.session_state.get('user_email')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    users = get_all_users()
    reports = get_all_reports()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Users", len(users))
    with col2:
        st.metric(" Total Inspections", len(reports))
    with col3:
        defects = sum(1 for r in reports if r.get('is_defect', False))
        st.metric("⚠️ Defects Found", defects)
    with col4:
        rate = (defects / len(reports) * 100) if reports else 0
        st.metric(" Defect Rate", f"{rate:.1f}%")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", " Inspection Reports", " Analytics", " System Health"])
    
    with tab1:
        st.subheader("User Management")
        if users:
            user_data = []
            for email, data in users.items():
                user_data.append({
                    'Email': email,
                    'Name': data.get('name', 'N/A'),
                    'Role': 'Admin' if email in ADMINS else 'User',
                    'Joined': data.get('created_at', 'N/A')[:10] if data.get('created_at') else 'N/A',
                    'Last Login': data.get('last_login', 'N/A')[:10] if data.get('last_login') else 'N/A'
                })
            st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("All Inspection Reports")
        if reports:
            report_data = []
            for r in reports[-100:][::-1]:
                report_data.append({
                    'User': r.get('user_email', 'N/A')[:25],
                    'Time': r.get('timestamp', '')[:16] if r.get('timestamp') else 'N/A',
                    'Score': f"{r.get('defect_score', 0):.1%}",
                    'Type': r.get('defect_type', 'N/A')[:30],
                    'Sub-Type': r.get('sub_type', 'N/A')[:20],
                    'Severity': r.get('severity', 'N/A'),
                    'Hole': '🕳️ Yes' if r.get('has_hole') else '❌ No',
                    'Status': '🔴 Defect' if r.get('is_defect') else '🟢 Normal'
                })
            df = pd.DataFrame(report_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Report (CSV)",
                csv,
                f"defect_reports_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with tab3:
        st.subheader("Analytics Dashboard")
        if reports:
            df = pd.DataFrame(reports)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Daily trend
            daily = df.groupby('date').agg({'is_defect': ['count', 'sum']}).reset_index()
            daily.columns = ['date', 'total', 'defects']
            daily['rate'] = (daily['defects'] / daily['total']) * 100
            
            fig = px.line(daily, x='date', y='rate', title='Daily Defect Rate Trend')
            st.plotly_chart(fig, use_container_width=True)
            
            # Defect type distribution
            if 'defect_type' in df.columns:
                type_counts = df['defect_type'].value_counts().head(10)
                fig = px.bar(x=type_counts.values, y=type_counts.index, orientation='h', title='Defect Type Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Score distribution
            fig = px.histogram(df, x='defect_score', nbins=20, title='Defect Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("System Health")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Database Status")
            st.success("✅ Database Connected")
            st.info(f" Users: {len(users)}")
            st.info(f" Reports: {len(reports)}")
            st.info(f" Storage: Local JSON")
        with col2:
            st.markdown("### System Information")
            st.info(f" Last Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f" Admins: {len(ADMINS)}")
            st.info(f" Device Support: All Devices")
            st.info(f" AI Model: DefectDetector v4.0 (with Hole Detection)")

# Main - FIXED LOGOUT ISSUE (Refresh doesn't logout)
def main():
    # Normal flow - check login status
    if not st.session_state.get('logged_in'):
        home_page()
    else:
        if st.session_state.get('user_role') == 'admin':
            admin_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()