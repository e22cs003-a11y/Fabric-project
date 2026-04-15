# app.py - Textile Defect Detection with Isolation Forest AI
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

# Scikit-learn imports for Isolation Forest
try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ scikit-learn or joblib not installed. Install with: pip install scikit-learn joblib")

# Page configuration
st.set_page_config(
    page_title="Textile Defect Detection System | AI-Powered",
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
CLIENT_ID = "933775442031-8q1kjhsanunatshkm6cb220ekvardovb.apps.googleusercontent.com"
CLIENT_SECRET = "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"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
REDIRECT_URI = "https://fabric-project-csc.streamlit.app/"

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
if 'isolation_model' not in st.session_state:
    st.session_state['isolation_model'] = None
if 'feature_scaler' not in st.session_state:
    st.session_state['feature_scaler'] = None

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

# Load Isolation Forest Model
def load_isolation_model():
    """Load the trained Isolation Forest model and scaler"""
    if not ML_AVAILABLE:
        return None, None
    
    model_path = 'models/isolation_forest.joblib'
    scaler_path = 'models/scaler.joblib'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.warning(f"⚠️ Could not load model: {e}")
            return None, None
    else:
        st.info("📁 AI Model not found. Run 'python train_model.py' first!")
        return None, None

def extract_texture_features_from_array(img_array):
    """
    Extract texture features from image array (for uploaded images)
    Features: Mean, Standard Deviation, Entropy, Edge Density, Skewness, Kurtosis
    """
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Resize to standard size for consistent features
        gray_resized = cv2.resize(gray, (128, 128))
        gray_normalized = gray_resized.astype(np.float32) / 255.0
        
        # Feature 1: Mean (average brightness)
        mean_val = np.mean(gray_normalized)
        
        # Feature 2: Standard Deviation (texture roughness)
        std_val = np.std(gray_normalized)
        
        # Feature 3: Entropy (measure of randomness/texture complexity)
        hist, _ = np.histogram(gray_normalized, bins=256, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist + 1e-10  # Avoid log(0)
        entropy_val = -np.sum(hist * np.log2(hist))
        
        # Feature 4: Edge density (for thread-out detection)
        edges = cv2.Canny(gray_resized, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Feature 5: Skewness (asymmetry of intensity distribution)
        skewness_val = np.mean(((gray_normalized - mean_val) / (std_val + 1e-10)) ** 3)
        
        # Feature 6: Kurtosis (peakedness of distribution)
        kurtosis_val = np.mean(((gray_normalized - mean_val) / (std_val + 1e-10)) ** 4) - 3
        
        return np.array([
            mean_val, std_val, entropy_val, edge_density, skewness_val, kurtosis_val
        ])
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

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

# Defect Detector with Isolation Forest
class IsolationForestDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['Mean', 'Std_Dev', 'Entropy', 'Edge_Density', 'Skewness', 'Kurtosis']
    
    def load_model(self):
        """Load the trained Isolation Forest model"""
        if ML_AVAILABLE:
            self.model, self.scaler = load_isolation_model()
            return self.model is not None
        return False
    
    def detect_holes(self, image):
        """Detect holes in the fabric using computer vision"""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use adaptive thresholding for better hole detection
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hole_detected = False
        hole_areas = []
        hole_positions = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 80:  # Filter small noise
                hole_detected = True
                hole_areas.append(area)
                x, y, w, h = cv2.boundingRect(cnt)
                hole_positions.append((x, y, w, h))
        
        return hole_detected, hole_areas, hole_positions
    
    def detect_thread_out(self, gray_image):
        """Detect thread-out/loose threads using edge detection"""
        edges = cv2.Canny(gray_image, 30, 100)
        
        # Look for linear patterns (threads)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        thread_out_detected = False
        thread_positions = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 40:  # Significant thread length
                    thread_out_detected = True
                    thread_positions.append((x1, y1, x2, y2))
        
        return thread_out_detected, thread_positions
    
    def detect_stains(self, gray_image, original_image):
        """Detect stains using intensity variation"""
        # Use local thresholding to find abnormal regions
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, local_thresh = cv2.threshold(blurred, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find stain regions
        contours, _ = cv2.findContours(local_thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        stain_detected = False
        stain_positions = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Stain-sized regions
                stain_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                stain_positions.append((x, y, w, h))
        
        return stain_detected, stain_positions
    
    def classify_defect(self, features, has_hole, has_thread_out, has_stain):
        """Classify defect type based on features and visual detection"""
        
        # Priority order: Hole > Thread Out > Stain > General Anomaly
        if has_hole:
            return ("Critical Hole / Tear", "Fabric Damage", "Critical", 95, "#dc3545")
        elif has_thread_out:
            return ("Thread Out / Loose Yarn", "Weave Irregularity", "High", 75, "#fd7e14")
        elif has_stain:
            return ("Oil / Chemical Stain", "Chemical Contamination", "Medium", 60, "#ffc107")
        else:
            # Use feature analysis for general anomaly
            anomaly_score = abs(features[1])  # Std deviation anomaly
            if anomaly_score > 2.5:
                return ("Severe Defect", "Major Anomaly", "High", 85, "#dc3545")
            elif anomaly_score > 1.5:
                return ("Minor Defect", "Surface Irregularity", "Medium", 55, "#ffc107")
            else:
                return ("Suspicious Area", "Needs Review", "Low", 40, "#fd7e14")
    
    def draw_detections(self, image, hole_positions, thread_positions, stain_positions):
        """Draw bounding boxes for all detected defects"""
        img_copy = np.array(image).copy()
        if len(img_copy.shape) == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        
        # Draw holes (RED)
        for (x, y, w, h) in hole_positions:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img_copy, "HOLE", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw thread-out (BLUE)
        for (x1, y1, x2, y2) in thread_positions:
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img_copy, "THREAD OUT", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw stains (GREEN)
        for (x, y, w, h) in stain_positions:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_copy, "STAIN", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy
    
    def analyze_fabric(self, image):
        """Main analysis function using Isolation Forest"""
        try:
            start_time = time.time()
            
            # Check if model is loaded
            if self.model is None:
                if not self.load_model():
                    return self.get_fallback_analysis(image)
            
            # Convert image to array
            img_array = np.array(image)
            
            # Extract features
            features = extract_texture_features_from_array(img_array)
            
            if features is None:
                return self.get_fallback_analysis(image)
            
            # Scale features and predict
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            anomaly_score = self.model.decision_function(features_scaled)[0]
            
            # Get raw feature values for analysis
            mean_val, std_val, entropy_val, edge_density, skewness, kurtosis = features
            
            # Determine if defect exists
            is_defect = (prediction == -1)
            
            # Computer vision detection for specific defect types
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            has_hole, hole_areas, hole_positions = self.detect_holes(image)
            has_thread_out, thread_positions = self.detect_thread_out(gray)
            has_stain, stain_positions = self.detect_stains(gray, image)
            
            # Classify defect type
            defect_name, sub_type, severity, severity_score, color = self.classify_defect(
                features, has_hole, has_thread_out, has_stain
            )
            
            # Calculate defect score from anomaly score (map -1 to 1 range to 0-1)
            # anomaly_score is negative for anomalies, positive for normal
            defect_score = 1.0 / (1.0 + np.exp(anomaly_score * 5)) if is_defect else 0.1
            
            # Calculate confidence
            confidence = 0.95 if is_defect else 0.90
            if has_hole:
                confidence = 0.98
            elif has_thread_out:
                confidence = 0.92
            
            # Create annotated image
            annotated_image = self.draw_detections(image, hole_positions, thread_positions, stain_positions)
            
            # Generate explanation
            explanation = self.generate_explanation(defect_name, has_hole, has_thread_out, has_stain, features)
            
            # Get recommendations
            actions, causes, prevention = self.get_recommendations(defect_name, has_hole, has_thread_out, has_stain)
            
            processing_time = round(time.time() - start_time, 2)
            
            return {
                'is_defect': is_defect or has_hole or has_thread_out or has_stain,
                'defect_score': float(defect_score),
                'anomaly_score': float(anomaly_score),
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
                'has_hole': has_hole,
                'has_thread_out': has_thread_out,
                'has_stain': has_stain,
                'hole_areas': hole_areas,
                'annotated_image': annotated_image,
                'features': {
                    'mean': float(mean_val),
                    'std_dev': float(std_val),
                    'entropy': float(entropy_val),
                    'edge_density': float(edge_density),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis)
                },
                'explanation': explanation
            }
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return self.get_fallback_analysis(image)
    
    def generate_explanation(self, defect_type, has_hole, has_thread_out, has_stain, features):
        """Generate human-readable explanation"""
        mean_val, std_val, entropy_val, edge_density, skewness, kurtosis = features
        
        explanation = "**🔍 AI Detection Analysis**\n\n"
        
        if has_hole:
            explanation += """
**🕳️ Hole/Tear Detected**

The Isolation Forest model identified a hole because:
- The image shows a complete break in fabric structure
- Pixel intensity drops to near-zero in affected area
- Contour analysis confirmed missing fabric region

**Action Required:** STOP production immediately!
"""
        elif has_thread_out:
            explanation += f"""
**🧵 Thread Out / Loose Yarn Detected**

The system detected thread irregularity because:
- Edge density ({edge_density:.3f}) indicates abnormal thread patterns
- Linear features suggest loose or broken threads
- Texture standard deviation ({std_val:.3f}) shows weaving inconsistency

**Recommended:** Inspect loom/knitting machine settings
"""
        elif has_stain:
            explanation += f"""
**💧 Chemical Stain Detected**

The AI identified a stain because:
- Mean brightness ({mean_val:.3f}) shows localized color variation
- Texture entropy ({entropy_val:.3f}) differs from normal fabric
- Affected area has different reflective properties

**Recommended:** Identify contamination source immediately
"""
        elif features[1] > 0.15:  # High std deviation
            explanation += f"""
**⚠️ Surface Irregularity Detected**

The Isolation Forest flagged this as anomalous because:
- Texture roughness (Std Dev: {std_val:.3f}) is higher than normal
- Normal fabric has more uniform texture distribution
- This could indicate weaving or knitting defects

**Recommended:** Schedule quality inspection
"""
        else:
            explanation += """
**✅ No Defects Detected**

The fabric passed all quality checks:
- Texture pattern matches normal fabric
- No holes, stains, or thread irregularities found
- Quality standards are met

**Status:** Accept for production
"""
        
        # Add feature comparison
        explanation += f"""

**📊 Feature Analysis:**
| Feature | Value | Status |
|---------|-------|--------|
| Mean Brightness | {mean_val:.3f} | {'✅ Normal' if 0.3 < mean_val < 0.7 else '⚠️ Check'} |
| Texture Std Dev | {std_val:.3f} | {'✅ Normal' if std_val < 0.15 else '⚠️ High'} |
| Texture Entropy | {entropy_val:.3f} | {'✅ Normal' if entropy_val < 5 else '⚠️ Complex'} |
| Edge Density | {edge_density:.3f} | {'✅ Normal' if edge_density < 0.15 else '⚠️ High'} |
"""
        
        return explanation
    
    def get_recommendations(self, defect_type, has_hole, has_thread_out, has_stain):
        """Get recommendations based on defect type"""
        if has_hole:
            actions = [
                "🛑 STOP production line immediately",
                "📌 Mark and isolate defective section",
                "🔧 Inspect machine for sharp edges/needles",
                "📋 Document defect for quality team",
                "🔄 Replace damaged parts before resuming"
            ]
            causes = [
                "Sharp object contact with fabric",
                "Machine needle damage or misalignment",
                "Excessive tension on fabric",
                "Foreign object in production line"
            ]
            prevention = [
                "Daily machine inspection checklist",
                "Use protective covers for sharp edges",
                "Regular maintenance schedule",
                "Install metal detectors on line"
            ]
        elif has_thread_out:
            actions = [
                "🔍 Inspect knitting/weaving machine settings",
                "📊 Check yarn tension consistency",
                "🔄 Realign thread feeding mechanism",
                "📝 Log defect for pattern analysis"
            ]
            causes = [
                "Uneven yarn tension",
                "Worn out machine needles",
                "Yarn quality variation",
                "Improper machine calibration"
            ]
            prevention = [
                "Regular yarn quality checks",
                "Scheduled needle replacement",
                "Tension monitoring system",
                "Operator training programs"
            ]
        elif has_stain:
            actions = [
                "🧪 Identify contamination source",
                "🧼 Clean affected machine parts",
                "🔬 Test fabric for chemical residue",
                "📊 Review lubrication procedures"
            ]
            causes = [
                "Oil leakage from machine",
                "Chemical spill during processing",
                "Dirty rollers or guides",
                "Improper cleaning procedures"
            ]
            prevention = [
                "Regular machine cleaning schedule",
                "Use food-grade lubricants",
                "Install drip trays",
                "Staff training on spill management"
            ]
        else:
            actions = [
                "✅ Fabric quality is acceptable",
                "📈 Continue with production",
                "🎯 Quality standards met",
                "📊 Record for quality tracking"
            ]
            causes = ["Normal fabric texture variation"]
            prevention = ["Continue current quality practices"]
        
        return actions, causes, prevention
    
    def get_fallback_analysis(self, image):
        """Fallback when model is not available"""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Basic detection
        std_dev = np.std(gray)
        has_hole, _, _ = self.detect_holes(image)
        has_thread_out, _ = self.detect_thread_out(gray)
        has_stain, _ = self.detect_stains(gray, image)
        
        is_defect = has_hole or has_thread_out or has_stain or std_dev > 50
        
        return {
            'is_defect': is_defect,
            'defect_score': 0.5 if is_defect else 0.1,
            'anomaly_score': 0,
            'defect_type': "Suspicious Area" if is_defect else "No Defect",
            'sub_type': "Manual Check Required" if is_defect else "Normal",
            'severity': "Unknown" if is_defect else "Good",
            'severity_score': 50 if is_defect else 0,
            'color': "#fd7e14" if is_defect else "#28a745",
            'actions': ["Please train the AI model for accurate detection"],
            'causes': ["Model not trained"],
            'prevention': ["Run python train_model.py"],
            'confidence': 0.5,
            'processing_time': 0.3,
            'has_hole': has_hole,
            'has_thread_out': has_thread_out,
            'has_stain': has_stain,
            'hole_areas': [],
            'annotated_image': np.array(image),
            'features': {'std_dev': float(std_dev)},
            'explanation': "⚠️ AI Model not trained. Run 'python train_model.py' with defect-free images."
        }

# Home Page with OAuth
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
        <div class="hero-title">✨ AI-Powered Defect Detection ✨</div>
        <div class="hero-subtitle">Powered by Isolation Forest Machine Learning<br>Detect holes, stains, and thread-outs instantly</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center; margin: 20px 0;"><h3>🔐 Sign In to Continue</h3></div>', unsafe_allow_html=True)
        
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
            <div class="feature-icon">🌲</div>
            <div class="feature-title">Isolation Forest AI</div>
            <div class="feature-desc">Lightweight ML model for fast anomaly detection</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🕳️</div>
            <div class="feature-title">Hole Detection</div>
            <div class="feature-desc">Specialized algorithm to detect holes and tears</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧵</div>
            <div class="feature-title">Thread-Out Detection</div>
            <div class="feature-desc">Identifies loose threads and knitting errors</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💧</div>
            <div class="feature-title">Stain Detection</div>
            <div class="feature-desc">Detects chemical and oil stains on fabric</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Mobile Ready</div>
            <div class="feature-desc">Works perfectly on all devices</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Analytics Dashboard</div>
            <div class="feature-desc">Track defects, trends and quality metrics</div>
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

# User Dashboard
def user_dashboard():
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <span style="font-size: 28px;">🧵</span>
            <span style="font-weight: 700; font-size: 20px;">AI Defect Detection Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    col_logout1, col_logout2 = st.columns([3, 1])
    with col_logout2:
        if st.button("🚪 Logout", key="logout_user", use_container_width=True):
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
    
    # Welcome Message with AI Status
    model_exists = os.path.exists('models/isolation_forest.joblib')
    model_status = "✅ AI Model Active" if model_exists else "⚠️ AI Model Not Trained"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                padding: 20px; border-radius: 16px; margin-bottom: 20px;">
        <h2>Welcome back, {st.session_state.get('user_name')}! 🎯</h2>
        <p style="color: #666;">Isolation Forest AI for fabric defect detection. {model_status}</p>
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
        st.markdown("### 📸 Fabric Inspection")
        
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
                if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
                    st.session_state['current_image'] = captured_image
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ Clear Image", use_container_width=True):
                    st.session_state['current_image'] = None
                    st.session_state['last_analysis'] = None
                    st.rerun()
    
    with col_right:
        st.markdown("### 📊 Quick Stats")
        
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
        st.markdown("## 🔬 AI Analysis Results")
        
        with st.spinner("🌲 Isolation Forest AI is analyzing fabric quality..."):
            detector = IsolationForestDetector()
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
                'anomaly_score': float(result.get('anomaly_score', 0)),
                'is_defect': bool(result['is_defect']),
                'defect_type': str(result['defect_type']),
                'sub_type': str(result['sub_type']),
                'severity': str(result['severity']),
                'confidence': float(result['confidence']),
                'processing_time': float(result['processing_time']),
                'has_hole': bool(result.get('has_hole', False)),
                'has_thread_out': bool(result.get('has_thread_out', False)),
                'has_stain': bool(result.get('has_stain', False))
            }
            save_report(report_data)
            
            # Display Original vs Annotated
            st.markdown("### 🖼️ Visual Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📷 Original Image**")
                st.image(st.session_state['current_image'], use_column_width=True)
            
            with col2:
                st.markdown("**🎯 Defect Detection View**")
                if result['annotated_image'] is not None:
                    st.image(result['annotated_image'], use_column_width=True)
                    
                    # Show detection summary
                    detections = []
                    if result.get('has_hole', False):
                        detections.append(f"🕳️ {len(result.get('hole_areas', []))} holes")
                    if result.get('has_thread_out', False):
                        detections.append("🧵 Thread out detected")
                    if result.get('has_stain', False):
                        detections.append("💧 Stain detected")
                    
                    if detections:
                        st.caption(" | ".join(detections))
                    elif result['is_defect']:
                        st.caption("⚠️ General anomaly detected")
                    else:
                        st.caption("✅ No defects detected")
                else:
                    st.image(st.session_state['current_image'], use_column_width=True)
            
            # Results Card
            if result['is_defect']:
                if result.get('has_hole', False):
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>🕳️ {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p style="font-size: 18px;">⚠️ CRITICAL: Hole/Tear detected in fabric!</p>
                        <p>AI Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("🚨 **URGENT: Hole/Tear detected. Stop machine immediately!**")
                elif result.get('has_thread_out', False):
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>🧵 {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p>AI Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ **Thread out detected. Inspect machine settings!**")
                elif result.get('has_stain', False):
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>💧 {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p>AI Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("🔍 **Stain detected. Check for contamination sources.**")
                else:
                    st.markdown(f"""
                    <div class="defect-card pulse">
                        <h2>⚠️ {result['defect_type']}</h2>
                        <p style="font-size: 16px;">Sub-Type: {result['sub_type']}</p>
                        <p>AI Confidence: {result['confidence']:.1%} | Severity: {result['severity']}</p>
                        <p>Defect Score: {result['defect_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-card pulse">
                    <h2>✅ {result['defect_type']}</h2>
                    <p style="font-size: 18px;">Quality Score: {(1-result['defect_score']):.1%}</p>
                    <p>AI Confidence: {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            
            # AI Explanation
            st.markdown("### 💡 AI Analysis Explanation")
            st.markdown(result.get('explanation', 'No explanation available'), unsafe_allow_html=True)
            
            # Metrics Row
            st.markdown("### 📊 Quality Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
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
            
            with metric_col5:
                anomaly_val = result.get('anomaly_score', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{anomaly_val:.2f}</div>
                    <div class="metric-label">Anomaly Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature Visualization
            if result.get('features'):
                st.markdown("### 📈 Texture Feature Analysis")
                features = result['features']
                
                # Create bar chart of features
                feature_df = pd.DataFrame({
                    'Feature': ['Mean', 'Std Dev', 'Entropy', 'Edge Density', 'Skewness', 'Kurtosis'],
                    'Value': [features['mean'], features['std_dev'], features['entropy'], 
                             features['edge_density'], features['skewness'], features['kurtosis']]
                })
                
                fig = px.bar(feature_df, x='Feature', y='Value', 
                            title='Fabric Texture Features',
                            color='Value', color_continuous_scale='RdYlGn')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Gauge Chart
            st.markdown("### 🎯 Defect Probability Gauge")
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
                    'anomaly_score': result.get('anomaly_score', 0),
                    'confidence': result['confidence'],
                    'has_hole': result.get('has_hole', False),
                    'has_thread_out': result.get('has_thread_out', False),
                    'has_stain': result.get('has_stain', False),
                    'features': result.get('features', {}),
                    'explanation': result.get('explanation', ''),
                    'timestamp': datetime.now().isoformat()
                }, indent=2)
                
                st.download_button(
                    label="📥 Download Full Report (JSON)",
                    data=report_json,
                    file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # New Inspection button
        if st.button("🔄 New Inspection", use_container_width=True):
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
    
    # Logout button
    col_logout1, col_logout2 = st.columns([3, 1])
    with col_logout2:
        if st.button("🚪 Logout", key="logout_admin", use_container_width=True):
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
        st.metric("📊 Total Inspections", len(reports))
    with col3:
        defects = sum(1 for r in reports if r.get('is_defect', False))
        st.metric("⚠️ Defects Found", defects)
    with col4:
        rate = (defects / len(reports) * 100) if reports else 0
        st.metric("📈 Defect Rate", f"{rate:.1f}%")
    
    # AI Model Status
    model_exists = os.path.exists('models/isolation_forest.joblib')
    if model_exists:
        st.success("✅ **AI Model Status:** Isolation Forest is trained and ready")
        st.info("🌲 Model Type: Isolation Forest | Features: Mean, Std Dev, Entropy, Edge Density, Skewness, Kurtosis")
    else:
        st.warning("⚠️ **AI Model Status:** Not trained. Run `python train_model.py` with defect-free images")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "📋 Inspection Reports", "📊 Analytics", "⚙️ System Health"])
    
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
                    'Thread': '🧵 Yes' if r.get('has_thread_out') else '❌ No',
                    'Stain': '💧 Yes' if r.get('has_stain') else '❌ No',
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
            
            # Defect categories over time
            if 'has_hole' in df.columns:
                df['hole_count'] = df['has_hole'].astype(int)
                df['thread_count'] = df['has_thread_out'].astype(int)
                df['stain_count'] = df['has_stain'].astype(int)
                
                daily_defects = df.groupby('date')[['hole_count', 'thread_count', 'stain_count']].sum().reset_index()
                fig = px.line(daily_defects, x='date', y=['hole_count', 'thread_count', 'stain_count'],
                             title='Defect Types Over Time',
                             labels={'value': 'Count', 'variable': 'Defect Type'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Score distribution
            fig = px.histogram(df, x='defect_score', nbins=20, title='Defect Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("System Health")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💾 Database Status")
            st.success("✅ Database Connected")
            st.info(f"👥 Users: {len(users)}")
            st.info(f"📊 Reports: {len(reports)}")
            st.info(f"💿 Storage: Local JSON")
        with col2:
            st.markdown("### 🌲 AI System")
            if model_exists:
                st.success("✅ Isolation Forest Model Loaded")
                st.info(f"🎯 Model: Isolation Forest (Scikit-learn)")
                st.info(f"📏 Features: 6 texture features")
                st.info(f"🌳 Trees: 100 estimators")
            else:
                st.warning("⚠️ Model Not Trained")
                st.info("Run: python train_model.py")
            st.info(f"📅 Last Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f"👑 Admins: {len(ADMINS)}")

# Main
def main():
    # Load isolation forest model into session state on startup
    if ML_AVAILABLE and st.session_state.get('isolation_model') is None:
        model, scaler = load_isolation_model()
        st.session_state['isolation_model'] = model
        st.session_state['feature_scaler'] = scaler
    
    if not st.session_state.get('logged_in'):
        home_page()
    else:
        if st.session_state.get('user_role') == 'admin':
            admin_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()
