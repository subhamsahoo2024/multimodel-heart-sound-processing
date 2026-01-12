"""
Multimodal Heart Disease Detection System
ECG (1D-CNN) + PCG (Gradient Boosting) Fusion with Late Decision-Level Fusion
================================================================================
Input Specifications:
- ECG: CSV with 187 signal values (single row or column)
- PCG: WAV audio file of heart sound
================================================================================
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="Multimodal Heart Disease Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .title-banner {
        background: linear-gradient(135deg, #0a3d62, #0c8fc1);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.2em;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        color: black;
    }
    .risk-high {
        background-color: #fdeaea;
        border-left: 6px solid #e74c3c;
    }
    .risk-low {
        background-color: #e8f6ef;
        border-left: 6px solid #2ecc71;
    }
    .risk-unknown {
        background-color: #eaf4fb;
        border-left: 6px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL PATHS & CONFIGURATION
# ============================================================================
MODEL_DIR = "./heart_sound_models"
PCG_MODEL_PATH = os.path.join(MODEL_DIR, "heart_sound_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
PCG_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

ECG_MODEL_DIR = "./heart_ecg_model"
ECG_MODEL_PATH = os.path.join(ECG_MODEL_DIR, "ecg_model_final.keras")

# Fusion weights (Late Fusion - Decision Level)
ECG_WEIGHT = 0.5  # Weight for ECG abnormality probability
PCG_WEIGHT = 0.5  # Weight for PCG abnormality probability
RISK_THRESHOLD = 0.5  # Threshold for HIGH/LOW risk classification


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_models():
    """Load both ECG and PCG models with error handling"""
    pcg_model, le, pcg_scaler = None, None, None
    ecg_model = None
    
    # Load PCG Model (Gradient Boosting/ML)
    if os.path.exists(PCG_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(PCG_SCALER_PATH):
        try:
            pcg_model = joblib.load(PCG_MODEL_PATH)
            le = joblib.load(LABEL_ENCODER_PATH)
            pcg_scaler = joblib.load(PCG_SCALER_PATH)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading PCG model: {e}")

    # Load ECG Model (1D-CNN)
    if os.path.exists(ECG_MODEL_PATH):
        try:
            ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading ECG model: {e}")
            
    return pcg_model, le, pcg_scaler, ecg_model


# ============================================================================
# ECG PREPROCESSING & NORMALIZATION
# ============================================================================
def preprocess_ecg(file) -> np.ndarray:
    """
    Preprocess ECG CSV file for 1D-CNN model
    
    Requirements:
    - Input: CSV with 187 ECG samples (row or column)
    - Output: (1, 187, 1) shaped array
    - Normalization: Min-Max scaling (0-1)
    
    Returns: Reshaped and normalized ECG data or None
    """
    try:
        # Read CSV file
        df = pd.read_csv(file, header=None)
        data = df.values.flatten().astype(np.float32)
        
        # Ensure exactly 187 samples
        REQUIRED_LENGTH = 187
        if len(data) > REQUIRED_LENGTH:
            # Take first 187 samples (centered around R-peak)
            data = data[:REQUIRED_LENGTH]
        elif len(data) < REQUIRED_LENGTH:
            # Pad with zeros if too short
            data = np.pad(data, (0, REQUIRED_LENGTH - len(data)), mode='constant', constant_values=0)
        
        # ====================================================================
        # NORMALIZATION: Min-Max Scaling (0-1)
        # ====================================================================
        # The 1D-CNN was trained on normalized data matching training distribution
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val > 1e-6:  # Avoid division by near-zero
            data_normalized = (data - min_val) / (max_val - min_val)
        else:
            # If constant signal, set to 0.5 (middle of 0-1 range)
            data_normalized = np.full_like(data, 0.5, dtype=np.float32)
        
        # ====================================================================
        # RESHAPE for 1D-CNN: (Batch_Size, Time_Steps, Channels)
        # ====================================================================
        # From (187,) to (1, 187, 1)
        ecg_reshaped = data_normalized.reshape(1, 187, 1)
        
        return ecg_reshaped
        
    except Exception as e:
        st.error(f"❌ ECG Preprocessing Error: {e}")
        return None


# ============================================================================
# PCG PREPROCESSING
# ============================================================================
def extract_features(audio_data, sr):
    """
    Extract MFCC and spectral features from audio
    Matches the feature extraction logic from model_training.py
    
    Returns: Feature vector or None
    """
    try:
        # Limit to 10 seconds
        if len(audio_data) > 10 * sr:
            audio_data = audio_data[:10 * sr]
        
        y = audio_data
        
        # 1. MFCCs (Mean & Variance)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_var = np.var(mfccs.T, axis=0)
        
        # 2. Delta MFCCs (Temporal Dynamics)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs.T, axis=0)
        delta_var = np.var(delta_mfccs.T, axis=0)
        
        # 3. Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(spectral_centroid)
        cent_var = np.var(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(spectral_bandwidth)
        bw_var = np.var(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_var = np.var(spectral_rolloff)
        
        # 4. Zero Crossing Rate (Noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # Concatenate all features
        features = np.concatenate([
            mfccs_mean, mfccs_var,
            delta_mean, delta_var,
            [cent_mean, cent_var, bw_mean, bw_var, rolloff_mean, rolloff_var, zcr_mean, zcr_var]
        ])
        return features
        
    except Exception as e:
        st.error(f"❌ PCG Feature Extraction Error: {e}")
        return None


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_ecg(ecg_model, ecg_data: np.ndarray) -> float:
    """
    Predict abnormality probability from ECG using 1D-CNN
    
    Args:
        ecg_model: Loaded Keras model
        ecg_data: Preprocessed ECG array (1, 187, 1)
        
    Returns: Probability of abnormality (0-1)
    """
    try:
        # 1D-CNN outputs sigmoid probability of abnormality
        prediction = ecg_model.predict(ecg_data, verbose=0)
        prob_abnormal = float(prediction[0][0])
        return prob_abnormal
    except Exception as e:
        st.error(f"❌ ECG Prediction Error: {e}")
        return None


def predict_pcg(pcg_model, le, pcg_scaler, features: np.ndarray) -> float:
    """
    Predict abnormality probability from PCG using Gradient Boosting/ML model
    
    Args:
        pcg_model: Loaded Scikit-learn model
        le: Label encoder
        pcg_scaler: Feature scaler
        features: Extracted feature vector (1, n_features)
        
    Returns: Probability of abnormality (0-1)
    """
    try:
        # Scale features
        features_scaled = pcg_scaler.transform(features)
        
        # Get probabilities for all classes
        probs = pcg_model.predict_proba(features_scaled)[0]
        
        # Find probability of "Abnormal" class
        if "Abnormal" in le.classes_:
            abnormal_idx = np.where(le.classes_ == "Abnormal")[0][0]
            prob_abnormal = probs[abnormal_idx]
        elif "Normal" in le.classes_:
            # If only "Normal" is labeled, Abnormal = 1 - Normal (binary)
            normal_idx = np.where(le.classes_ == "Normal")[0][0]
            prob_abnormal = 1.0 - probs[normal_idx]
        else:
            # Fallback: assume last class is abnormal
            prob_abnormal = probs[-1] if len(probs) > 1 else probs[0]
            
        return prob_abnormal
        
    except Exception as e:
        st.error(f"❌ PCG Prediction Error: {e}")
        return None


# ============================================================================
# FUSION & DIAGNOSIS
# ============================================================================
def calculate_combined_risk(ecg_prob: float, pcg_prob: float) -> float:
    """
    Late Fusion: Weighted Average of Probabilities
    
    Formula: Combined_Risk = (w_ecg * ECG_Prob) + (w_pcg * PCG_Prob)
    
    Args:
        ecg_prob: ECG abnormality probability
        pcg_prob: PCG abnormality probability
        
    Returns: Combined risk score (0-1)
    """
    combined = (ECG_WEIGHT * ecg_prob) + (PCG_WEIGHT * pcg_prob)
    return combined


def get_diagnosis(risk_score: float) -> tuple:
    """
    Determine diagnosis based on risk score
    
    Returns: (diagnosis_label, risk_level, color_class)
    """
    if risk_score > RISK_THRESHOLD:
        return "HIGH RISK - Abnormality Detected", "HIGH", "risk-high"
    else:
        return "LOW RISK - Within Normal Range", "LOW", "risk-low"


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_ecg_waveform(ecg_data: np.ndarray):
    """Plot the normalized ECG waveform"""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ecg_data[0, :, 0], linewidth=2, color="#0c8fc1")
    ax.set_title("Preprocessed ECG Signal (187 samples, Normalized)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized Amplitude (0-1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_pcg_spectrogram(audio_data: np.ndarray, sr: int):
    """Plot spectrogram of heart sound"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Compute STFT and convert to dB
    stft = np.abs(librosa.stft(audio_data))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz", cmap="viridis", ax=ax)
    ax.set_title("Heart Sound Spectrogram (PCG)", fontsize=12, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)


def plot_probability_comparison(ecg_prob: float, pcg_prob: float, combined_prob: float):
    """Plot comparison of probabilities"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    categories = ["ECG\n(1D-CNN)", "PCG\n(Gradient Boosting)", "Combined\n(Fusion)"]
    probabilities = [ecg_prob * 100, pcg_prob * 100, combined_prob * 100]
    colors = ["#3498db" if p <= 50 else "#e74c3c" for p in probabilities]
    
    bars = ax.bar(categories, probabilities, color=colors, edgecolor="black", linewidth=2)
    ax.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Risk Threshold (50%)")
    ax.set_ylabel("Abnormality Probability (%)", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f"{prob:.1f}%", ha="center", va="bottom", fontweight="bold")
    
    st.pyplot(fig)


# ============================================================================
# SIDEBAR STATUS
# ============================================================================
def render_sidebar():
    """Render model status in sidebar"""
    st.sidebar.markdown("### 📊 System Status")
    
    pcg_model, le, pcg_scaler, ecg_model = load_models()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if ecg_model:
            st.sidebar.success("✅ ECG Model")
        else:
            st.sidebar.error("❌ ECG Model")
            
    with col2:
        if pcg_model:
            st.sidebar.success("✅ PCG Model")
        else:
            st.sidebar.error("❌ PCG Model")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Fusion Configuration")
    st.sidebar.markdown(f"**ECG Weight:** {ECG_WEIGHT * 100:.0f}%")
    st.sidebar.markdown(f"**PCG Weight:** {PCG_WEIGHT * 100:.0f}%")
    st.sidebar.markdown(f"**Risk Threshold:** {RISK_THRESHOLD * 100:.0f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Input Specifications")
    st.sidebar.markdown("""
    **ECG:**
    - Format: CSV or TXT
    - Length: 187 samples
    - Values: Raw or normalized
    
    **PCG:**
    - Format: WAV audio
    - Duration: 5-10 seconds
    - Quality: 16 kHz+ recommended
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application flow"""
    
    # Render sidebar
    render_sidebar()
    
    # Title banner
    st.markdown("""
    <div class="title-banner">
        <h1>🫀 Multimodal Heart Disease Detection</h1>
        <p>Real-time fusion of ECG (1D-CNN) and PCG (Gradient Boosting) analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    pcg_model, le, pcg_scaler, ecg_model = load_models()
    
    # Check if at least one model is available
    if pcg_model is None and ecg_model is None:
        st.error("❌ Both models failed to load. Please check model paths and files.")
        return
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    st.markdown("### 📁 Upload Your Data")
    
    col_ecg, col_pcg = st.columns(2)
    
    with col_ecg:
        st.markdown("#### 📊 ECG Signal Input")
        st.markdown("Expected: CSV file with **187 heartbeat samples**")
        uploaded_ecg = st.file_uploader("Choose ECG file (.csv, .txt)", type=["csv", "txt"], key="ecg")
        
    with col_pcg:
        st.markdown("#### 🔊 Heart Sound Input")
        st.markdown("Expected: WAV audio file of heart sound")
        uploaded_pcg = st.file_uploader("Choose PCG file (.wav)", type=["wav"], key="pcg")
    
    # ========================================================================
    # ANALYSIS BUTTON
    # ========================================================================
    st.markdown("")
    if st.button("🔬 Run Multimodal Analysis", type="primary", use_container_width=True):
        
        if not uploaded_ecg and not uploaded_pcg:
            st.error("❌ Please upload at least one file (ECG or PCG)")
            return
        
        ecg_prob = None
        pcg_prob = None
        ecg_signal = None
        pcg_audio = None
        
        # ====================================================================
        # PROCESS ECG
        # ====================================================================
        if uploaded_ecg and ecg_model:
            with st.spinner("⏳ Processing ECG..."):
                ecg_data = preprocess_ecg(uploaded_ecg)
                
                if ecg_data is not None:
                    ecg_prob = predict_ecg(ecg_model, ecg_data)
                    ecg_signal = ecg_data
                    st.success("✅ ECG analysis complete")
        elif uploaded_ecg:
            st.warning("⚠️ ECG model not loaded")
        
        # ====================================================================
        # PROCESS PCG
        # ====================================================================
        if uploaded_pcg and pcg_model:
            with st.spinner("⏳ Processing PCG..."):
                try:
                    y, sr = librosa.load(uploaded_pcg, sr=22050)
                    features = extract_features(y, sr)
                    
                    if features is not None:
                        features = features.reshape(1, -1)
                        pcg_prob = predict_pcg(pcg_model, le, pcg_scaler, features)
                        pcg_audio = y
                        st.success("✅ PCG analysis complete")
                except Exception as e:
                    st.error(f"❌ PCG Processing Error: {e}")
        elif uploaded_pcg:
            st.warning("⚠️ PCG model not loaded")
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📈 Signal Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            if ecg_signal is not None:
                st.markdown("#### ECG Waveform")
                plot_ecg_waveform(ecg_signal)
        
        with viz_col2:
            if pcg_audio is not None:
                st.markdown("#### PCG Spectrogram")
                plot_pcg_spectrogram(pcg_audio, 22050)
        
        # ====================================================================
        # RESULTS SECTION
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📋 Diagnostic Results")
        
        if ecg_prob is not None and pcg_prob is not None:
            # Both models available: Use fusion
            combined_prob = calculate_combined_risk(ecg_prob, pcg_prob)
            diagnosis, risk_level, color_class = get_diagnosis(combined_prob)
            
            # Display individual model scores
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("ECG Risk Score", f"{ecg_prob*100:.1f}%")
            
            with res_col2:
                st.metric("PCG Risk Score", f"{pcg_prob*100:.1f}%")
            
            with res_col3:
                st.metric("Combined Risk", f"{combined_prob*100:.1f}%")
            
            # Display final diagnosis
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h3>🎯 Final Multimodal Diagnosis</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Combined Risk Score:</strong> {combined_prob*100:.2f}%</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability comparison chart
            st.markdown("")
            plot_probability_comparison(ecg_prob, pcg_prob, combined_prob)
            
        elif ecg_prob is not None:
            # Only ECG available
            diagnosis, risk_level, color_class = get_diagnosis(ecg_prob)
            
            st.metric("ECG Risk Score", f"{ecg_prob*100:.1f}%")
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h3>ECG-Only Diagnosis (PCG unavailable)</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Risk Score:</strong> {ecg_prob*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif pcg_prob is not None:
            # Only PCG available
            diagnosis, risk_level, color_class = get_diagnosis(pcg_prob)
            
            st.metric("PCG Risk Score", f"{pcg_prob*100:.1f}%")
            st.markdown(f"""
            <div class="metric-card {color_class} text-blue-700">
                <h3>PCG-Only Diagnosis (ECG unavailable)</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Risk Score:</strong> {pcg_prob*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ No valid predictions generated. Please check your input files.")


if __name__ == "__main__":
    main()