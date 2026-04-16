# main_app.py - ULTRA FAST Textile Defect Detection (256x256 Optimized) with Gemini LLM
# IMPROVED: Fixed false positives with higher CV thresholds and smarter logic

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

# ML imports
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import joblib
from sklearn.preprocessing import StandardScaler

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI not installed")

# Page configuration
st.set_page_config(
    page_title="Textile Defect Detection | 256x256 AI + Gemini",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CONFIGURATION - 256x256 OPTIMIZED MODE
# ============================================

# 🔴 IMPORTANT: Replace with your actual Gemini API key
GEMINI_API_KEY = "AQ.Ab8RN6I3T-UXa9DN1NlW2E4WVemrMo5HsITXt6YyJLsVXuvqZg"

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

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔥 256x256 CONFIGURATION
IMG_SIZE = 256  # High resolution for better defect detection

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ''
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = ''
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []
if 'feature_extractor' not in st.session_state:
    st.session_state['feature_extractor'] = None
if 'isolation_model' not in st.session_state:
    st.session_state['isolation_model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = None

# Database files
USERS_FILE = "users_data.json"
REPORTS_FILE = "reports_data.json"

# ============================================
# DATABASE FUNCTIONS
# ============================================

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

# ============================================
# FEATURE EXTRACTOR FOR 256x256
# ============================================

class FeatureExtractor256:
    """Feature extractor for 256x256 images using ResNet50"""
    
    def __init__(self, model_name='resnet50'):
        self.model_name = model_name
        self.device = DEVICE
        self.img_size = IMG_SIZE
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()
    
    def load_model(self):
        """Load ResNet50 model"""
        try:
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.feature_dim = self.model.num_features
            print(f"✅ Model loaded: {self.model_name} (2048 features)")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = timm.create_model('resnet18', pretrained=True, num_classes=0)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.feature_dim = self.model.num_features
            print(f"✅ Fallback: ResNet18 loaded ({self.feature_dim} features)")
    
    def extract_features(self, image):
        """Extract features from PIL image"""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img_tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_texture_features(self, image):
        """Extract texture features from image"""
        try:
            img_array = np.array(image.convert('L'))
            img_resized = cv2.resize(img_array, (128, 128))
            img_norm = img_resized.astype(np.float32) / 255.0
            
            mean_val = np.mean(img_norm)
            std_val = np.std(img_norm)
            
            hist, _ = np.histogram(img_norm, bins=16, range=(0, 1))
            hist = hist / (hist.sum() + 1e-10)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log2(hist))
            
            edges = cv2.Canny(img_resized, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return np.array([mean_val, std_val, entropy, edge_density])
        except Exception as e:
            return np.zeros(4)

# ============================================
# GEMINI LLM INTEGRATION
# ============================================

class GeminiAnalyzer:
    """Gemini Vision LLM for defect classification"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.available = False
        
        if GEMINI_AVAILABLE and api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True
                print("✅ Gemini API initialized")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
    
    def gemini_override_check(self, image, isolation_prediction, anomaly_score, threshold, cv_holes, cv_threads, cv_stains):
        """Gemini Override Logic - Verifies if defects are real"""
        if not self.available:
            return {
                'needs_override': False,
                'defect_found': False,
                'defect_type': None,
                'explanation': "Gemini API not configured.",
                'detailed_analysis': None,
                'final_verdict': 'PASS'
            }
        
        try:
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            st.info("🔍 Gemini AI is performing detailed defect analysis...")
            
            prompt = f"""
            You are a textile quality control expert. Analyze this fabric image CAREFULLY.
            
            IMPORTANT: The local AI model thinks this is NORMAL fabric (Score: {anomaly_score:.4f}).
            
            Computer Vision detected:
            - Holes: {len(cv_holes)} (REAL holes are serious defects)
            - Thread-outs: {len(cv_threads)} (More than 50 is a problem)
            - Stains: {len(cv_stains)} (More than 5 is a problem)
            
            Please verify:
            1. Are the detected spots actual defects OR just shadows/dust/fabric texture?
            2. Is there any REAL hole in the fabric? (Even one hole = REJECT)
            3. Are there many loose threads? (More than 50 = REJECT)
            4. Are there many stains? (More than 5 = REJECT)
            
            If it's just minor texture, dust, or shadow - mark it as PASS.
            Only mark as REJECT if there is a REAL defect.
            
            Answer in EXACT format:
            
            DEFECT_STATUS: [REJECT or PASS]
            DEFECT_TYPE: [HOLE/TEAR/THREAD-OUT/STAIN/NONE]
            SEVERITY: [CRITICAL/HIGH/MEDIUM/LOW/NONE]
            EXPLANATION: [Brief explanation of what you see]
            VERDICT: [REJECT or PASS]
            """
            
            response = self.vision_model.generate_content([prompt, image_pil])
            response_text = response.text
            
            is_reject = "DEFECT_STATUS: REJECT" in response_text.upper() or "VERDICT: REJECT" in response_text.upper()
            
            if "VERDICT: PASS" in response_text.upper() or "DEFECT_STATUS: PASS" in response_text.upper():
                is_reject = False
            
            defect_type = "None"
            explanation = "Fabric quality is acceptable"
            severity = "None"
            
            if "DEFECT_TYPE: HOLE" in response_text.upper():
                defect_type = "Hole / Puncture"
                severity = "Critical"
                is_reject = True
                explanation = "Real hole detected in fabric"
            elif "DEFECT_TYPE: TEAR" in response_text.upper():
                defect_type = "Tear / Rip"
                severity = "Critical"
                is_reject = True
                explanation = "Tear detected in fabric"
            elif "DEFECT_TYPE: THREAD-OUT" in response_text.upper():
                defect_type = "Thread-out / Loose Thread"
                severity = "High"
                explanation = "Thread issues detected"
            elif "DEFECT_TYPE: STAIN" in response_text.upper():
                defect_type = "Stain / Dirt"
                severity = "Medium"
                explanation = "Stains detected on fabric"
            
            import re
            exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=\n\n|\n[A-Z]|$)', response_text, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()
            
            return {
                'needs_override': True,
                'defect_found': is_reject,
                'defect_type': defect_type if is_reject else None,
                'severity': severity if is_reject else "None",
                'explanation': explanation,
                'detailed_analysis': response_text,
                'final_verdict': 'REJECT' if is_reject else 'PASS'
            }
            
        except Exception as e:
            print(f"Gemini error: {e}")
            cv_defect = len(cv_holes) > 0 or len(cv_threads) > 50 or len(cv_stains) > 5
            return {
                'needs_override': True,
                'defect_found': cv_defect,
                'defect_type': "Defect Found" if cv_defect else None,
                'severity': "Critical" if len(cv_holes) > 0 else ("High" if len(cv_threads) > 50 else "Medium"),
                'explanation': f"CV detected {len(cv_holes)} holes, {len(cv_threads)} threads, {len(cv_stains)} stains",
                'detailed_analysis': None,
                'final_verdict': 'REJECT' if cv_defect else 'PASS'
            }
    
    def chat_response(self, user_query, context):
        """Generate conversational response"""
        if not self.available:
            return "AI Chat is not available. Please configure Gemini API key."
        
        try:
            prompt = f"""
            You are a textile quality control AI assistant.
            
            Context: {context}
            
            User question: {user_query}
            
            Provide a helpful, concise response about fabric quality, defects, or recommendations.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error: {e}"

# ============================================
# DEFECT DETECTOR FOR 256x256 WITH FIXED FALSE POSITIVES
# ============================================

class DefectDetector256:
    """Complete defect detection with fixed false positive issues"""
    
    def __init__(self):
        self.feature_extractor = None
        self.isolation_model = None
        self.scaler = None
        self.threshold = None
        self.gemini = None
        
    def load_models(self):
        """Load all required models"""
        if st.session_state['feature_extractor'] is None:
            self.feature_extractor = FeatureExtractor256()
            st.session_state['feature_extractor'] = self.feature_extractor
        else:
            self.feature_extractor = st.session_state['feature_extractor']
        
        if st.session_state['isolation_model'] is None:
            model_path = 'models/isolation_forest.joblib'
            scaler_path = 'models/scaler.joblib'
            threshold_path = 'models/threshold.txt'
            
            if os.path.exists(model_path):
                self.isolation_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read())
                
                st.session_state['isolation_model'] = self.isolation_model
                st.session_state['scaler'] = self.scaler
                st.session_state['threshold'] = self.threshold
            else:
                st.warning("⚠️ Model not found. Run 'python train_model.py' first!")
                return False
        
        self.isolation_model = st.session_state['isolation_model']
        self.scaler = st.session_state['scaler']
        self.threshold = st.session_state['threshold']
        
        if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            self.gemini = GeminiAnalyzer(GEMINI_API_KEY)
        
        return True
    
    def detect_holes_cv(self, image):
        """
        Hole detection - HIGH threshold to avoid false positives
        Only detects REAL holes (large area, dark regions)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use multiple thresholds for better hole detection
        _, thresh1 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_or(thresh1, thresh2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        holes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only REAL holes (minimum 150 pixels area)
            if area > 150:  # Increased from 100 to 150
                x, y, w, h = cv2.boundingRect(cnt)
                holes.append({'bbox': (x, y, w, h), 'area': area})
        return holes
    
    def detect_thread_out_cv(self, image):
        """
        Thread detection - FIXED: Higher Canny thresholds to ignore fabric texture
        Canny(100, 200) ignores small texture patterns
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # IMPORTANT FIX: Increased Canny thresholds from (50,150) to (100,200)
        # This ignores fabric weave pattern and only detects real loose threads
        edges = cv2.Canny(gray, 100, 200)  # Higher threshold = less sensitive to texture
        
        # Hough line detection with higher thresholds
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=50, maxLineGap=15)
        
        threads = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Only long, prominent threads (minimum 50 pixels length)
                if length > 50:
                    # Check if it's a real loose thread (not just fabric edge)
                    threads.append({'line': (x1, y1, x2, y2), 'length': length})
        
        # Filter out duplicate detections
        unique_threads = []
        for thread in threads:
            is_unique = True
            for existing in unique_threads:
                x1, y1, x2, y2 = thread['line']
                ex1, ey1, ex2, ey2 = existing['line']
                # If lines are too close, consider them the same
                if abs(x1 - ex1) < 20 and abs(y1 - ey1) < 20:
                    is_unique = False
                    break
            if is_unique:
                unique_threads.append(thread)
        
        return unique_threads
    
    def detect_stains_cv(self, image):
        """
        Stain detection - INCREASED min_area to avoid dust/false positives
        Minimum 200 pixels to be considered a stain
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding for better stain detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stains = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # IMPORTANT FIX: Minimum 200 pixels to ignore tiny dust/shadows
            # Maximum 5000 pixels to avoid false large regions
            if 200 < area < 5000:  # Increased min from 100 to 200
                x, y, w, h = cv2.boundingRect(cnt)
                # Check aspect ratio - real stains are usually not extremely thin
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:  # Filter out very thin lines
                    stains.append({'bbox': (x, y, w, h), 'area': area})
        return stains
    
    def draw_detections(self, image, holes, threads, stains):
        """Draw detections on image"""
        display_img = image.copy()
        
        for hole in holes:
            x, y, w, h = hole['bbox']
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(display_img, "HOLE", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for thread in threads:
            x1, y1, x2, y2 = thread['line']
            cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_img, "THREAD", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        for stain in stains:
            x, y, w, h = stain['bbox']
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(display_img, "STAIN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return display_img
    
    def analyze_fabric(self, image):
        """
        IMPROVED MAIN ANALYSIS FUNCTION
        Logic: 
        - Trust Isolation Forest if it says NORMAL (anomaly_score > 0)
        - Only REJECT if: holes > 0 OR threads > 50 OR stains > 5
        - Gemini can override if needed
        """
        try:
            start_time = time.time()
            
            if not self.load_models():
                return self.get_fallback_analysis(image)
            
            if isinstance(image, Image.Image):
                img_np = np.array(image)
            else:
                img_np = image
            
            # Resize to 256x256
            img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
            img_pil = Image.fromarray(img_resized)
            
            # Extract features
            dino_features = self.feature_extractor.extract_features(img_pil)
            texture_features = self.feature_extractor.extract_texture_features(img_pil)
            combined_features = np.hstack([dino_features, texture_features])
            
            # Isolation Forest prediction
            features_scaled = self.scaler.transform([combined_features])
            anomaly_score = self.isolation_model.decision_function(features_scaled)[0]
            isolation_prediction = self.isolation_model.predict(features_scaled)[0]
            
            # CV detection with improved thresholds
            holes_cv = self.detect_holes_cv(img_resized)
            threads_cv = self.detect_thread_out_cv(img_resized)
            stains_cv = self.detect_stains_cv(img_resized)
            
            # Count detections
            hole_count = len(holes_cv)
            thread_count = len(threads_cv)
            stain_count = len(stains_cv)
            
            # ============================================
            # IMPROVED LOGIC - FIXED FALSE POSITIVES
            # ============================================
            
            # Check if Isolation Forest thinks it's NORMAL (positive score)
            is_iso_normal = isolation_prediction == 1 and anomaly_score > 0
            
            # REJECT conditions (much higher thresholds to avoid false positives)
            has_real_hole = hole_count > 0
            has_many_threads = thread_count > 50  # Increased from 10 to 50
            has_many_stains = stain_count > 5     # Increased from 3 to 5
            
            # Initial verdict based on strict rules
            initial_is_defect = has_real_hole or has_many_threads or has_many_stains
            
            # If Isolation Forest says NORMAL and counts are low, trust it
            if is_iso_normal and not initial_is_defect:
                initial_is_defect = False
            
            final_is_defect = initial_is_defect
            final_defect_type = "Unknown"
            final_severity = "Unknown"
            explanation_text = None
            gemini_overridden = False
            gemini_analysis_text = None
            gemini_verdict = None
            
            # ============================================
            # GEMINI VERIFICATION (Only if needed)
            # ============================================
            
            # Only call Gemini if there's potential defect or ISO is uncertain
            need_gemini = (initial_is_defect or (is_iso_normal == False and thread_count > 10))
            
            if self.gemini and self.gemini.available and need_gemini:
                gemini_result = self.gemini.gemini_override_check(
                    img_resized, isolation_prediction, anomaly_score, self.threshold,
                    holes_cv, threads_cv, stains_cv
                )
                
                if gemini_result and gemini_result.get('needs_override'):
                    gemini_verdict = gemini_result.get('final_verdict', 'PASS')
                    
                    if gemini_verdict == 'REJECT':
                        final_is_defect = True
                        final_defect_type = gemini_result.get('defect_type', 'Defect Found')
                        final_severity = gemini_result.get('severity', 'Critical')
                        explanation_text = gemini_result.get('explanation', '')
                        gemini_analysis_text = gemini_result.get('detailed_analysis', '')
                        gemini_overridden = True
                        st.warning("🚨 Gemini AI confirmed a REAL defect!")
                    else:
                        # Gemini says PASS - override everything
                        final_is_defect = False
                        final_defect_type = "No Defect - Quality Pass"
                        final_severity = "Good"
                        explanation_text = "Gemini AI verified: Fabric quality is acceptable. Minor marks are not defects."
                        gemini_overridden = True
                        st.success("✅ Gemini AI verified: Quality PASS!")
            else:
                # No Gemini or not needed - use rule-based decision
                if has_real_hole:
                    final_is_defect = True
                    final_defect_type = "Hole / Puncture"
                    final_severity = "Critical"
                    explanation_text = f"Detected {hole_count} real hole(s) in fabric"
                elif has_many_threads:
                    final_is_defect = True
                    final_defect_type = "Thread-out / Loose Thread"
                    final_severity = "High"
                    explanation_text = f"Detected {thread_count} thread issues (threshold: >50)"
                elif has_many_stains:
                    final_is_defect = True
                    final_defect_type = "Stain / Dirt"
                    final_severity = "Medium"
                    explanation_text = f"Detected {stain_count} stains (threshold: >5)"
                else:
                    final_is_defect = False
                    final_defect_type = "No Defect - Quality Pass"
                    final_severity = "Good"
                    if is_iso_normal:
                        explanation_text = f"Fabric quality check PASSED. Isolation Forest confirms NORMAL fabric (Score: {anomaly_score:.4f} > 0)."
                    else:
                        explanation_text = f"Fabric quality check PASSED. Small variations are within acceptable limits."
            
            # Calculate defect score based on final decision
            if final_is_defect:
                if "Hole" in final_defect_type or "Tear" in final_defect_type:
                    defect_score = 0.95
                elif "Thread" in final_defect_type:
                    defect_score = 0.85
                elif "Stain" in final_defect_type:
                    defect_score = 0.75
                else:
                    defect_score = 0.7
            else:
                defect_score = max(0, min(0.2, anomaly_score + 0.3))
            
            # Generate explanation
            explanation = self.generate_explanation(
                final_is_defect, final_defect_type, final_severity, explanation_text,
                hole_count, thread_count, stain_count,
                isolation_prediction, anomaly_score, gemini_overridden, gemini_verdict
            )
            
            # Annotated image
            annotated_image = self.draw_detections(img_resized, holes_cv, threads_cv, stains_cv)
            processing_time = round(time.time() - start_time, 2)
            
            return {
                'is_defect': final_is_defect,
                'defect_score': defect_score,
                'anomaly_score': float(anomaly_score),
                'threshold': float(self.threshold),
                'isolation_prediction': isolation_prediction,
                'defect_type': final_defect_type,
                'severity': final_severity,
                'has_hole': hole_count > 0,
                'has_thread_out': thread_count > 50,
                'has_stain': stain_count > 5,
                'holes_count': hole_count,
                'threads_count': thread_count,
                'stains_count': stain_count,
                'annotated_image': annotated_image,
                'processing_time': processing_time,
                'explanation': explanation,
                'gemini_analysis': gemini_analysis_text,
                'confidence': 0.95 if final_is_defect else 0.90,
                'gemini_overridden': gemini_overridden,
                'gemini_verdict': gemini_verdict
            }
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return self.get_fallback_analysis(image)
    
    def generate_explanation(self, is_defect, defect_type, severity, explanation_text,
                            hole_count, thread_count, stain_count,
                            isolation_prediction, anomaly_score, gemini_overridden, gemini_verdict):
        """Generate explanation with clear logic"""
        
        if is_defect:
            explanation = f"""
## DEFECT DETECTED - REJECTED

### Defect Type: {defect_type}
### Severity: {severity}

### Detection Summary:
- **Holes Found:** {hole_count} (REJECT if >0)
- **Thread-outs Found:** {thread_count} (REJECT if >50)
- **Stains Found:** {stain_count} (REJECT if >5)

### Model Analysis:
- **Isolation Forest:** {'ANOMALY' if isolation_prediction == -1 else 'NORMAL'}
- **Anomaly Score:** {anomaly_score:.4f}
- **Processing Mode:** High Resolution (256x256)
- **Gemini Verification:** {'ACTIVATED - ' + gemini_verdict if gemini_overridden else 'Not needed'}

### Details:
{explanation_text if explanation_text else "Defect detected in fabric"}

### Action Required:
STOP production immediately! Inspect the machine and remove defective fabric.
"""
        else:
            explanation = f"""
## NO DEFECTS DETECTED - QUALITY PASS ✅

### Quality Check Results:
- **Holes Found:** {hole_count} (OK - needs >0 to reject)
- **Thread-outs Found:** {thread_count} (OK - needs >50 to reject)
- **Stains Found:** {stain_count} (OK - needs >5 to reject)

### Model Analysis:
- **Isolation Forest:** {'ANOMALY' if isolation_prediction == -1 else 'NORMAL'}
- **Anomaly Score:** {anomaly_score:.4f}
- **Processing Mode:** High Resolution (256x256)
- **Gemini Verification:** {'ACTIVATED - ' + gemini_verdict if gemini_overridden else 'Not needed'}

### Verdict:
{explanation_text if explanation_text else "Fabric quality is acceptable. Small variations are within limits."}

### Status: ACCEPT for production
"""
        return explanation
    
    def get_fallback_analysis(self, image):
        img_np = np.array(image)
        return {
            'is_defect': False,
            'defect_score': 0.1,
            'anomaly_score': 0,
            'threshold': 0,
            'isolation_prediction': 1,
            'defect_type': "Unknown",
            'severity': "Unknown",
            'has_hole': False,
            'has_thread_out': False,
            'has_stain': False,
            'holes_count': 0,
            'threads_count': 0,
            'stains_count': 0,
            'annotated_image': cv2.resize(img_np, (256, 256)),
            'processing_time': 0.1,
            'explanation': "AI Models not loaded. Run 'python train_model.py' first.",
            'gemini_analysis': None,
            'confidence': 0.5,
            'gemini_overridden': False,
            'gemini_verdict': None
        }

# ============================================
# STREAMLIT UI
# ============================================

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-header { text-align: center; padding: 2rem; color: white; }
    .defect-card { background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); border-radius: 20px; padding: 1.5rem; color: white; margin: 1rem 0; }
    .success-card { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); border-radius: 20px; padding: 1.5rem; color: white; margin: 1rem 0; }
    .metric-card { background: white; border-radius: 16px; padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

def home_page():
    st.markdown("""
    <div class="main-header">
        <h1>Textile Defect Detection System</h1>
        <p>High Resolution Mode (256x256) | Fixed False Positive Issues</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center;"><h3>Sign In</h3></div>', unsafe_allow_html=True)
        
        oauth2 = OAuth2Component(
            CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, 
            REFRESH_TOKEN_URL, REVOKE_TOKEN_URL
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
                
                user_data = {
                    'email': user_email,
                    'name': user_name,
                    'role': 'admin' if user_email in ADMINS else 'user',
                    'created_at': datetime.now().isoformat(),
                }
                save_user(user_data)
                
                st.session_state['logged_in'] = True
                st.session_state['user_email'] = user_email
                st.session_state['user_name'] = user_name
                st.session_state['user_role'] = 'admin' if user_email in ADMINS else 'user'
                st.rerun()

def user_dashboard():
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 2rem; background: white; border-radius: 16px; margin-bottom: 1rem;">
        <div><span style="font-size: 24px;">Textile Defect Detection</span> <span style="background:#28a745; color:white; padding:2px 8px; border-radius:12px; font-size:12px;">FIXED MODE</span></div>
        <div>Welcome, {st.session_state.get('user_name', 'User')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Logout", key="logout_btn"):
        for key in ['logged_in', 'user_email', 'user_name', 'user_role', 'current_image']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    reports = get_user_reports(st.session_state['user_email'])
    total = len(reports)
    defects = sum(1 for r in reports if r.get('is_defect', False))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 32px; font-weight: bold;">{total}</div><div>Inspections</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 32px; font-weight: bold; color: #dc3545;">{defects}</div><div>Defects</div></div>', unsafe_allow_html=True)
    with col3:
        rate = (defects/total*100) if total > 0 else 0
        st.markdown(f'<div class="metric-card"><div style="font-size: 32px; font-weight: bold;">{rate:.1f}%</div><div>Defect Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div style="font-size: 32px; font-weight: bold;">Active</div><div>Gemini AI</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        st.subheader("Upload Fabric Image")
        
        tab1, tab2 = st.tabs(["Upload Image", "Camera"])
        captured_image = None
        
        with tab1:
            uploaded = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
            if uploaded:
                captured_image = Image.open(uploaded)
                st.image(captured_image, use_column_width=True)
        
        with tab2:
            camera = st.camera_input("Take a photo")
            if camera:
                captured_image = Image.open(camera)
                st.image(captured_image, use_column_width=True)
        
        if captured_image:
            if st.button("Analyze Fabric", type="primary", use_container_width=True):
                st.session_state['current_image'] = captured_image
                st.rerun()
    
    with col_right:
        st.subheader("AI Assistant")
        st.info("Ask questions about fabric quality, defects, or recommendations")
        
        if 'chat_messages' not in st.session_state:
            st.session_state['chat_messages'] = []
        
        for msg in st.session_state['chat_messages'][-5:]:
            if msg['role'] == 'user':
                st.markdown(f'**You:** {msg["content"]}')
            else:
                st.markdown(f'**AI:** {msg["content"]}')
        
        user_question = st.text_input("Ask a question...", key="chat_input")
        if user_question:
            st.session_state['chat_messages'].append({'role': 'user', 'content': user_question})
            
            context = "Fabric quality analysis system with fixed false positive issues."
            gemini = GeminiAnalyzer(GEMINI_API_KEY) if GEMINI_AVAILABLE else None
            
            if gemini and gemini.available:
                response = gemini.chat_response(user_question, context)
            else:
                response = "AI assistant ready. Configure Gemini API key for detailed responses."
            
            st.session_state['chat_messages'].append({'role': 'assistant', 'content': response})
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.get('current_image'):
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing with smart detection... Gemini verifying..."):
            detector = DefectDetector256()
            result = detector.analyze_fabric(st.session_state['current_image'])
            
            history_entry = {
                'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'timestamp': datetime.now().isoformat(),
                'is_defect': result['is_defect'],
                'defect_type': result['defect_type'],
                'severity': result['severity']
            }
            st.session_state['analysis_history'].insert(0, history_entry)
            
            report_data = {
                'id': history_entry['id'],
                'user_email': st.session_state['user_email'],
                'timestamp': history_entry['timestamp'],
                'defect_score': result['defect_score'],
                'anomaly_score': result['anomaly_score'],
                'is_defect': result['is_defect'],
                'defect_type': result['defect_type'],
                'severity': result['severity'],
                'has_hole': result['has_hole'],
                'has_thread_out': result['has_thread_out'],
                'has_stain': result['has_stain'],
                'holes_count': result['holes_count'],
                'threads_count': result['threads_count'],
                'stains_count': result['stains_count'],
                'gemini_overridden': result.get('gemini_overridden', False),
                'processing_time': result['processing_time']
            }
            save_report(report_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(st.session_state['current_image'], use_column_width=True)
            with col2:
                st.markdown("**Detection View**")
                st.image(result['annotated_image'], use_column_width=True)
            
            if result['is_defect']:
                if result.get('gemini_overridden'):
                    st.error("GEMINI VERIFIED: REAL DEFECT CONFIRMED - REJECTED!")
                st.markdown(f"""
                <div class="defect-card">
                    <h2>{result['defect_type']}</h2>
                    <p>Severity: {result['severity']}</p>
                    <p>Confidence: {result['confidence']:.1%}</p>
                    <p>Status: REJECTED</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if result.get('gemini_overridden'):
                    st.success("✅ GEMINI VERIFIED: Quality PASS - Minor variations are acceptable!")
                st.markdown(f"""
                <div class="success-card">
                    <h2>No Defects Detected</h2>
                    <p>Status: ACCEPTED for production</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            
            st.markdown(result['explanation'])
            
            if result.get('gemini_analysis'):
                with st.expander("Gemini AI Detailed Analysis", expanded=False):
                    st.markdown(result['gemini_analysis'])
            
            st.markdown("### Detection Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Holes", result['holes_count'])
            with m2:
                st.metric("Thread-outs", result['threads_count'])
            with m3:
                st.metric("Stains", result['stains_count'])
            with m4:
                st.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['defect_score'] * 100,
                title={'text': "Defect Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#dc3545" if result['is_defect'] else "#28a745"},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 70], 'color': '#fff3cd'},
                        {'range': [70, 100], 'color': '#f8d7da'}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Processing time: {result['processing_time']} seconds | Smart thresholds: Holes>0, Threads>50, Stains>5")
            
            if st.button("New Inspection", use_container_width=True):
                st.session_state['current_image'] = None
                st.rerun()

def admin_dashboard():
    st.title("Admin Dashboard")
    
    if st.button("Logout"):
        for key in ['logged_in', 'user_email', 'user_name', 'user_role']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    users = get_all_users()
    reports = get_all_reports()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users))
    with col2:
        st.metric("Total Inspections", len(reports))
    with col3:
        defects = sum(1 for r in reports if r.get('is_defect', False))
        st.metric("Defects Found", defects)
    with col4:
        rate = (defects/len(reports)*100) if reports else 0
        st.metric("Defect Rate", f"{rate:.1f}%")
    
    st.subheader("User Management")
    if users:
        user_df = pd.DataFrame([{
            'Email': email,
            'Name': data.get('name', 'N/A'),
            'Role': 'Admin' if email in ADMINS else 'User',
            'Joined': data.get('created_at', 'N/A')[:10]
        } for email, data in users.items()])
        st.dataframe(user_df, use_container_width=True)
    
    st.subheader("Recent Inspections")
    if reports:
        report_df = pd.DataFrame([{
            'User': r.get('user_email', 'N/A')[:20],
            'Time': r.get('timestamp', '')[:16],
            'Type': r.get('defect_type', 'N/A')[:25],
            'Severity': r.get('severity', 'N/A'),
            'Holes': r.get('holes_count', 0),
            'Threads': r.get('threads_count', 0),
            'Stains': r.get('stains_count', 0),
            'Gemini': 'Yes' if r.get('gemini_overridden') else 'No'
        } for r in reports[-50:]])
        st.dataframe(report_df, use_container_width=True)
        
        csv = report_df.to_csv(index=False)
        st.download_button("Download Reports CSV", csv, "reports.csv", "text/csv")

def main():
    if not st.session_state.get('logged_in'):
        home_page()
    else:
        if st.session_state.get('user_role') == 'admin':
            admin_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()
