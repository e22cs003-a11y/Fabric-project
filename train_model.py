# train_model.py - Train Isolation Forest for Fabric Defect Detection
import os
import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = 128  # Resize images to this size for feature extraction

def extract_texture_features(image_path):
    """
    Extract texture features from fabric image
    Features: Mean, Standard Deviation, Entropy
    """
    try:
        # Load image in grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to standard size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Feature 1: Mean (average brightness)
        mean_val = np.mean(img_array)
        
        # Feature 2: Standard Deviation (texture roughness)
        std_val = np.std(img_array)
        
        # Feature 3: Entropy (measure of randomness/texture complexity)
        # Calculate histogram for entropy
        hist, _ = np.histogram(img_array, bins=256, range=(0, 1))
        hist = hist / hist.sum()  # Normalize
        # Avoid log(0)
        hist = hist + 1e-10
        entropy_val = -np.sum(hist * np.log2(hist))
        
        # Additional feature 4: Edge density (for thread-out detection)
        edges = cv2.Canny((img_array * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Additional feature 5: Skewness (asymmetry of intensity distribution)
        skewness_val = np.mean(((img_array - mean_val) / (std_val + 1e-10)) ** 3)
        
        # Additional feature 6: Kurtosis (peakedness of distribution)
        kurtosis_val = np.mean(((img_array - mean_val) / (std_val + 1e-10)) ** 4) - 3
        
        return [
            mean_val,           # 0: Average brightness
            std_val,            # 1: Texture roughness
            entropy_val,        # 2: Texture complexity
            edge_density,       # 3: Edge concentration
            skewness_val,       # 4: Distribution asymmetry
            kurtosis_val        # 5: Distribution peakedness
        ]
    
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None

def load_defect_free_features(folder_path):
    """
    Load all defect-free images and extract features
    """
    features = []
    image_paths = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        print(f"📁 Creating folder: {folder_path}")
        os.makedirs(folder_path)
        print("\n❌ Please add defect-free fabric images to this folder and run again!")
        print("📝 Instructions:")
        print("   1. Take 50-100 photos of PERFECT fabric")
        print("   2. Save them as .jpg or .png in dataset/defect_free/")
        print("   3. Run this script again")
        return np.array([]), []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(valid_extensions)]
    
    if len(image_files) == 0:
        print(f"⚠️ No images found in {folder_path}")
        print("📁 Please add defect-free fabric images to train the model")
        return np.array([]), []
    
    print(f"📸 Found {len(image_files)} defect-free images")
    print("🔄 Extracting features from images...")
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        features_vec = extract_texture_features(img_path)
        
        if features_vec is not None:
            features.append(features_vec)
            image_paths.append(img_path)
            
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(image_files)} images")
    
    print(f"✅ Successfully extracted features from {len(features)} images")
    return np.array(features), image_paths

def train_isolation_forest():
    """
    Train Isolation Forest model for anomaly detection
    """
    print("=" * 60)
    print("🧵 Textile Defect Detection - Isolation Forest Training")
    print("=" * 60)
    print("\n📊 Feature Extraction:")
    print("   - Mean (average brightness)")
    print("   - Standard Deviation (texture roughness)")
    print("   - Entropy (texture complexity)")
    print("   - Edge Density (thread concentration)")
    print("   - Skewness (intensity asymmetry)")
    print("   - Kurtosis (distribution peakedness)")
    print()
    
    # Load features from defect-free images
    features, image_paths = load_defect_free_features('dataset/defect_free')
    
    if len(features) == 0:
        print("\n❌ Training failed: No valid images found!")
        print("\n📁 Please follow these steps:")
        print("   1. Create a folder named 'dataset/defect_free/'")
        print("   2. Add 50-100 defect-free fabric images")
        print("   3. Run this script again")
        return False
    
    print(f"\n📊 Feature matrix shape: {features.shape}")
    print(f"   Features per image: {features.shape[1]}")
    
    # Normalize features
    print("\n🔄 Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train Isolation Forest
    print("\n🚀 Training Isolation Forest model...")
    print("   Parameters:")
    print("   - contamination: 0.01 (expect 1% anomalies)")
    print("   - n_estimators: 100 trees")
    print("   - max_samples: 'auto'")
    
    iso_forest = IsolationForest(
        contamination=0.01,      # Expect 1% of images to be anomalies
        n_estimators=100,        # Number of trees
        max_samples='auto',      # Use all samples for training
        random_state=42,         # Reproducibility
        bootstrap=False,         # No bootstrap sampling
        verbose=1
    )
    
    iso_forest.fit(features_scaled)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/isolation_forest.joblib'
    joblib.dump(iso_forest, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = 'models/scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Save feature names for reference
    feature_names = ['Mean', 'Std_Dev', 'Entropy', 'Edge_Density', 'Skewness', 'Kurtosis']
    with open('models/feature_names.txt', 'w') as f:
        f.write(','.join(feature_names))
    
    # Test on training data (should all be normal)
    predictions = iso_forest.predict(features_scaled)
    normal_count = np.sum(predictions == 1)
    anomaly_count = np.sum(predictions == -1)
    
    print("\n📊 Training Results:")
    print(f"   - Normal samples identified: {normal_count}/{len(features)}")
    print(f"   - Anomaly samples identified: {anomaly_count}/{len(features)}")
    
    if anomaly_count > 0:
        print(f"\n⚠️ Warning: {anomaly_count} training images were marked as anomalies!")
        print("   These images might have defects or be too different from normal fabric.")
        print("   Consider reviewing these images or adding more defect-free samples.")
    
    # Calculate average feature values for reference
    print("\n📈 Average Feature Values (Defect-Free Fabric):")
    avg_features = np.mean(features, axis=0)
    for i, name in enumerate(feature_names):
        print(f"   - {name}: {avg_features[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print("\n📁 Model files created:")
    print("   - models/isolation_forest.joblib (main model)")
    print("   - models/scaler.joblib (feature normalizer)")
    print("   - models/feature_names.txt (feature reference)")
    print("\n🚀 Next step: Run 'streamlit run main_app.py'")
    
    return True

def test_model():
    """
    Test the trained model with a sample image
    """
    print("\n" + "=" * 60)
    print("🧪 Testing the trained model")
    print("=" * 60)
    
    model_path = 'models/isolation_forest.joblib'
    scaler_path = 'models/scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Model not found. Please train first.")
        return
    
    # Load model and scaler
    iso_forest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: Isolation Forest")
    print(f"   - Number of trees: {iso_forest.n_estimators}")
    print(f"   - Contamination: {iso_forest.contamination}")
    
    # Test with a random defect-free image if available
    test_folder = 'dataset/defect_free'
    if os.path.exists(test_folder):
        test_files = [f for f in os.listdir(test_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if test_files:
            test_path = os.path.join(test_folder, test_files[0])
            features = extract_texture_features(test_path)
            
            if features:
                features_scaled = scaler.transform([features])
                prediction = iso_forest.predict(features_scaled)[0]
                score = iso_forest.decision_function(features_scaled)[0]
                
                print(f"\n📸 Test image: {test_files[0]}")
                print(f"   - Prediction: {'✅ NORMAL' if prediction == 1 else '⚠️ ANOMALY/DEFECT'}")
                print(f"   - Anomaly score: {score:.4f} (negative = anomaly)")
    
    print("\n✅ Model is ready for use in main_app.py!")

if __name__ == "__main__":
    success = train_isolation_forest()
    if success:
        test_model()
