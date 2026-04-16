# train_model.py - ULTRA OPTIMIZED 256x256 Training (FAST MODE)
# Complete in 25-30 minutes for 1177 images

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import timm
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ULTRA OPTIMIZED CONFIGURATION - 256x256 FAST
# ============================================

IMG_SIZE = 256                    # 256x256 resolution
BATCH_SIZE = 16                   # Smaller batch for 256x256 (memory efficient)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ESTIMATORS = 30                 # Very fast training (reduced from 100)
CONTAMINATION = 0.01              # 1% anomaly rate
USE_PATCHES = False               # Disable patches for speed

print("=" * 60)
print("⚡ ULTRA OPTIMIZED 256x256 TRAINING - FAST MODE")
print("=" * 60)
print(f"Resolution: {IMG_SIZE}x{IMG_SIZE} (65,536 pixels)")
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Estimators: {N_ESTIMATORS}")
print(f"Expected Time: 25-30 minutes for 1177 images")
print("=" * 60)

# ============================================
# FAST FEATURE EXTRACTOR FOR 256x256
# ============================================

class Fast256Extractor:
    """Ultra-fast feature extractor for 256x256 images"""
    
    def __init__(self):
        print("\n📦 Loading ResNet50...")
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.feature_dim = self.model.num_features  # 2048 features
        
        # Simple transform for 256x256
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        print(f"✅ ResNet50 loaded (2048 features)")
    
    def extract_batch(self, image_paths):
        """Extract features in BATCH - MUCH FASTER"""
        batch_tensors = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                batch_tensors.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                continue
        
        if not batch_tensors:
            return np.array([]), []
        
        # Stack and process in batch
        batch = torch.stack(batch_tensors).to(DEVICE)
        
        with torch.no_grad():
            features = self.model(batch)
        
        return features.cpu().numpy(), valid_paths

# ============================================
# FAST TEXTURE FEATURES (Optimized)
# ============================================

def get_texture_fast(image_path):
    """Ultra-fast texture extraction for 256x256"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(4)
        
        # Resize to 128x128 for faster texture extraction
        img = cv2.resize(img, (128, 128))
        img_norm = img.astype(np.float32) / 255.0
        
        # Simple statistics (fast)
        mean_val = np.mean(img_norm)
        std_val = np.std(img_norm)
        
        # Fast entropy
        hist, _ = np.histogram(img_norm, bins=16, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10) + 1e-10
        entropy = -np.sum(hist * np.log2(hist))
        
        # Edge density (for defect detection)
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return np.array([mean_val, std_val, entropy, edge_density])
    except:
        return np.zeros(4)

# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def train():
    start_time = time.time()
    
    # Step 1: Load all images
    print("\n📂 Loading images from dataset/defect_free/...")
    image_paths = []
    folder = 'dataset/defect_free'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"❌ Folder '{folder}' created. Please add 1177 images!")
        return False
    
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(folder, f))
    
    print(f"✅ Found {len(image_paths)} images")
    
    if len(image_paths) < 10:
        print("❌ Need at least 10 images in dataset/defect_free/")
        return False
    
    # Step 2: Initialize extractor
    extractor = Fast256Extractor()
    
    # Step 3: Extract DNN features in BATCHES (FAST)
    print("\n🎨 Extracting ResNet50 features from 256x256 images...")
    print("⏱️ This will take 15-20 minutes...")
    
    all_features = []
    valid_paths = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Batch processing"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_features, valid_batch = extractor.extract_batch(batch_paths)
        
        if len(batch_features) > 0:
            all_features.extend(batch_features)
            valid_paths.extend(valid_batch)
    
    all_features = np.array(all_features)
    print(f"\n✅ DNN features extracted!")
    print(f"   Shape: {all_features.shape}")
    print(f"   Dimension: {all_features.shape[1]}")
    
    # Step 4: Extract texture features (FAST)
    print("\n📊 Extracting texture features...")
    print("⏱️ This will take 3-5 minutes...")
    
    texture_features = []
    for path in tqdm(valid_paths, desc="Texture features"):
        tex = get_texture_fast(path)
        texture_features.append(tex)
    
    texture_features = np.array(texture_features)
    print(f"✅ Texture features extracted!")
    print(f"   Shape: {texture_features.shape}")
    
    # Step 5: Combine features
    print("\n🔗 Combining DNN + Texture features...")
    combined_features = np.hstack([all_features, texture_features])
    print(f"✅ Combined shape: {combined_features.shape}")
    
    # Step 6: Normalize
    print("\n📈 Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)
    print("✅ Normalization complete")
    
    # Step 7: Train Isolation Forest (VERY FAST with 30 estimators)
    print("\n🚀 Training Isolation Forest...")
    print(f"   Estimators: {N_ESTIMATORS}")
    print(f"   Contamination: {CONTAMINATION}")
    
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,           # Use all CPU cores
        verbose=0
    )
    
    iso_forest.fit(features_scaled)
    print("✅ Training complete!")
    
    # Step 8: Calculate threshold (1st percentile)
    print("\n🎯 Calculating anomaly threshold...")
    train_scores = iso_forest.decision_function(features_scaled)
    threshold = np.percentile(train_scores, 1)
    
    print(f"   Score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(f"   Mean score: {np.mean(train_scores):.4f}")
    print(f"   Threshold (1%): {threshold:.4f}")
    
    # Step 9: Save models
    print("\n💾 Saving models to 'models/' folder...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(iso_forest, 'models/isolation_forest.joblib', compress=3)
    print("✅ Saved: models/isolation_forest.joblib")
    
    joblib.dump(scaler, 'models/scaler.joblib', compress=3)
    print("✅ Saved: models/scaler.joblib")
    
    with open('models/threshold.txt', 'w') as f:
        f.write(str(threshold))
    print(f"✅ Saved: models/threshold.txt (value: {threshold:.4f})")
    
    # Save feature info
    feature_info = {
        'img_size': IMG_SIZE,
        'feature_dim': extractor.feature_dim,
        'texture_dim': 4,
        'total_dim': combined_features.shape[1],
        'threshold': float(threshold),
        'n_estimators': N_ESTIMATORS,
        'training_samples': len(valid_paths),
        'model_type': 'resnet50',
        'resolution': f'{IMG_SIZE}x{IMG_SIZE}'
    }
    
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print("✅ Saved: models/feature_info.json")
    
    # Step 10: Validation
    print("\n📊 Validation...")
    predictions = iso_forest.predict(features_scaled)
    anomalies = np.sum(predictions == -1)
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️ Total time: {minutes} minutes {seconds} seconds")
    print(f"📐 Resolution: {IMG_SIZE}x{IMG_SIZE} (65,536 pixels)")
    print(f"📊 Total images: {len(valid_paths)}")
    print(f"⚠️ Anomalies detected: {anomalies} ({anomalies/len(valid_paths)*100:.1f}%)")
    print(f"🎯 Threshold: {threshold:.4f}")
    
    print("\n📁 Model files saved in 'models/' folder:")
    print("   ✓ isolation_forest.joblib")
    print("   ✓ scaler.joblib")
    print("   ✓ threshold.txt")
    print("   ✓ feature_info.json")
    
    print("\n🚀 NEXT STEP:")
    print("   streamlit run main_app.py")
    
    print("=" * 60)
    
    return True

# ============================================
# RUN TRAINING
# ============================================

if __name__ == "__main__":
    print("\n⚡ ULTRA OPTIMIZED 256x256 TRAINING STARTED")
    print("🎯 Target: 25-30 minutes for 1177 images\n")
    
    success = train()
    
    if not success:
        print("\n❌ Training failed!")
        print("\nPlease ensure:")
        print("   1. Folder 'dataset/defect_free' exists")
        print("   2. Contains 1177 images")
        print("   3. Images are JPG/PNG format")
