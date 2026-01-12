"""
Multimodal Heart Disease Detection System
ECG (1D-CNN) + PCG (CRNN Deep Learning) Fusion with Late Decision-Level Fusion
================================================================================
Input Specifications:
- ECG: CSV with 187 signal values (single row or column)
- PCG: WAV audio file of heart sound (5 seconds)
================================================================================
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, filtfilt
from pathlib import Path

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
PCG_MODEL_PATH = os.path.join(MODEL_DIR, "pcg_crnn_model.keras")

ECG_MODEL_DIR = "./heart_ecg_model"
ECG_MODEL_PATH = os.path.join(ECG_MODEL_DIR, "ecg_model_final.keras")

# PCG Audio Configuration (Must match training)
SAMPLE_RATE = 1000  # Hz
DURATION = 5  # seconds
TARGET_LENGTH = 5000  # samples (SAMPLE_RATE * DURATION)
LOWCUT = 20  # Hz
HIGHCUT = 400  # Hz

# Fusion configuration
RISK_THRESHOLD = 0.5  # Threshold for HIGH/LOW risk classification


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_models():
    """Load both ECG and PCG models with error handling"""
    pcg_model = None
    ecg_model = None
    
    # Load PCG Model (CRNN Deep Learning)
    if os.path.exists(PCG_MODEL_PATH):
        try:
            pcg_model = tf.keras.models.load_model(PCG_MODEL_PATH)
            st.sidebar.success("✅ PCG CRNN Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading PCG model: {e}")

    # Load ECG Model (1D-CNN)
    if os.path.exists(ECG_MODEL_PATH):
        try:
            ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH)
            st.sidebar.success("✅ ECG CNN Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading ECG model: {e}")
            
    return pcg_model, ecg_model


# ============================================================================
# ECG PREPROCESSING & NORMALIZATION (KEPT EXACTLY AS ORIGINAL)
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
        ecg_reshaped = data_normalized.reshape(1, 187, 1)
        
        return ecg_reshaped
        
    except Exception as e:
        st.error(f"❌ ECG Preprocessing Error: {e}")
        return None


# ============================================================================
# PCG PREPROCESSING (NEW - MATCHES TRAINING PHASE)
# ============================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to remove noise"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_pcg(audio_file) -> tuple:
    """
    Preprocess PCG audio file using SLIDING WINDOW approach for long recordings
    
    Strategy:
    - Load entire audio file
    - Segment into 5-second chunks (0-5s, 5-10s, 10-15s...)
    - Preprocess each chunk (bandpass, normalize, reshape)
    - Return all chunks for batch prediction
    
    Returns: (preprocessed_chunks, num_chunks, chunk_info) or (None, 0, None)
    """
    try:
        # Step 1: Load entire audio at 1000 Hz
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        
        # Apply bandpass filter (20-400 Hz) to entire audio
        audio = butter_bandpass_filter(audio, LOWCUT, HIGHCUT, SAMPLE_RATE)
        
        # Step 2: Calculate number of 5-second chunks
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / TARGET_LENGTH))
        
        # Step 3: Segment and preprocess each chunk
        preprocessed_chunks = []
        chunk_info = []
        
        for i in range(num_chunks):
            start_idx = i * TARGET_LENGTH
            end_idx = min(start_idx + TARGET_LENGTH, total_samples)
            
            # Extract chunk
            chunk = audio[start_idx:end_idx]
            
            # Pad if last chunk is shorter than 5 seconds
            if len(chunk) < TARGET_LENGTH:
                padding = TARGET_LENGTH - len(chunk)
                chunk = np.pad(chunk, (0, padding), mode='constant')
            
            # Max-Abs Normalization (per chunk)
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val
            
            # Reshape for CRNN: (1, 5000, 1)
            chunk_reshaped = chunk.reshape(1, TARGET_LENGTH, 1)
            preprocessed_chunks.append(chunk_reshaped)
            
            # Store chunk timing info
            start_time = start_idx / SAMPLE_RATE
            end_time = end_idx / SAMPLE_RATE
            chunk_info.append((i+1, start_time, end_time))
        
        # Stack all chunks into batch: (num_chunks, 5000, 1)
        if preprocessed_chunks:
            batch_data = np.vstack(preprocessed_chunks)
            return batch_data, num_chunks, chunk_info
        else:
            return None, 0, None
        
    except Exception as e:
        st.error(f"❌ PCG Preprocessing Error: {e}")
        return None, 0, None


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


def predict_pcg(pcg_model, pcg_data: np.ndarray, num_chunks: int, chunk_info: list) -> tuple:
    """
    Predict abnormality using SLIDING WINDOW + MAX-RISK STRATEGY
    
    Medical Logic: If a murmur occurs in ANY 5-second window, patient is abnormal
    
    Args:
        pcg_model: Loaded Keras CRNN model
        pcg_data: Batch of preprocessed chunks (num_chunks, 5000, 1)
        num_chunks: Number of 5-second segments
        chunk_info: List of (chunk_num, start_time, end_time) tuples
        
    Returns: (max_probability, chunk_probabilities, highest_risk_chunk_idx) or (None, None, None)
    """
    try:
        # Batch prediction on all chunks
        predictions = pcg_model.predict(pcg_data, verbose=0)
        chunk_probabilities = predictions.flatten().tolist()
        
        # Max-Risk Aggregation: Take highest probability
        max_prob = float(np.max(predictions))
        max_idx = int(np.argmax(predictions))
        
        return max_prob, chunk_probabilities, max_idx
        
    except Exception as e:
        st.error(f"❌ PCG Prediction Error: {e}")
        return None, None, None


# ============================================================================
# FUSION & DIAGNOSIS
# ============================================================================
def calculate_combined_risk(ecg_prob: float, pcg_prob: float, fusion_method: str = "average") -> float:
    """
    Late Fusion: Combine ECG and PCG probabilities
    
    Methods:
    - average: (ECG_Prob + PCG_Prob) / 2
    - max: max(ECG_Prob, PCG_Prob) - More conservative, catches any abnormality
    
    Args:
        ecg_prob: ECG abnormality probability
        pcg_prob: PCG abnormality probability
        fusion_method: "average" or "max"
        
    Returns: Combined risk score (0-1)
    """
    if fusion_method == "max":
        # Use max for "Confused Case" - if either modality detects abnormality
        combined = max(ecg_prob, pcg_prob)
    else:
        # Default: Average (50/50 fusion)
        combined = (ecg_prob + pcg_prob) / 2.0
    
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


def plot_pcg_waveform(pcg_data: np.ndarray):
    """Plot the preprocessed PCG waveform"""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(pcg_data[0, :, 0], linewidth=1, color="#e74c3c")
    ax.set_title("Preprocessed PCG Signal (5000 samples @ 1000Hz, Max-Abs Normalized)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_pcg_spectrogram(audio_data: np.ndarray, sr: int):
    """Plot spectrogram of heart sound"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Flatten the audio data if it's 3D
    if len(audio_data.shape) == 3:
        audio_data = audio_data[0, :, 0]
    
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
    
    categories = ["ECG\n(1D-CNN)", "PCG\n(CRNN)", "Combined\n(Fusion)"]
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
def render_sidebar(fusion_method: str):
    """Render model status in sidebar"""
    st.sidebar.markdown("### 📊 System Status")
    
    pcg_model, ecg_model = load_models()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if ecg_model:
            st.sidebar.success("✅ ECG CNN")
        else:
            st.sidebar.error("❌ ECG CNN")
            
    with col2:
        if pcg_model:
            st.sidebar.success("✅ PCG CRNN")
        else:
            st.sidebar.error("❌ PCG CRNN")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Fusion Configuration")
    st.sidebar.markdown(f"**Method:** {fusion_method.upper()}")
    st.sidebar.markdown(f"**Risk Threshold:** {RISK_THRESHOLD * 100:.0f}%")
    
    if fusion_method == "average":
        st.sidebar.info("📊 Average Fusion: (ECG + PCG) / 2")
    else:
        st.sidebar.info("📊 Max Fusion: max(ECG, PCG)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Input Specifications")
    st.sidebar.markdown("""
    **ECG:**
    - Format: CSV or TXT
    - Length: 187 samples
    - Values: Raw or normalized
    
    **PCG:**
    - Format: WAV audio
    - Duration: ~5 seconds
    - Sample Rate: Resampled to 1000Hz
    - Preprocessing: Bandpass filtered (20-400Hz)
    """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application flow"""
    
    # Fusion method selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Fusion Method")
    fusion_method = st.sidebar.radio(
        "Select fusion strategy:",
        ["average", "max"],
        index=0,
        help="Average: Standard 50/50 fusion | Max: Conservative (detects any abnormality)"
    )
    
    # Render sidebar
    render_sidebar(fusion_method)
    
    # Title banner
    st.markdown("""
    <div class="title-banner">
        <h1>🫀 Multimodal Heart Disease Detection</h1>
        <p>Real-time fusion of ECG (1D-CNN) and PCG (CRNN Deep Learning) analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    pcg_model, ecg_model = load_models()
    
    # Check if at least one model is available
    if pcg_model is None and ecg_model is None:
        st.error("❌ Both models failed to load. Please check model paths and files.")
        st.info(f"Expected PCG model at: {PCG_MODEL_PATH}")
        st.info(f"Expected ECG model at: {ECG_MODEL_PATH}")
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
        st.markdown("Expected: WAV audio file of heart sound (~5 seconds)")
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
        pcg_signal = None
        
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
        # PROCESS PCG (SLIDING WINDOW + MAX-RISK STRATEGY)
        # ====================================================================
        pcg_chunks_info = None
        pcg_chunk_probs = None
        pcg_highest_risk_chunk = None
        
        if uploaded_pcg and pcg_model:
            with st.spinner("⏳ Processing PCG with sliding window analysis..."):
                try:
                    # Preprocess audio - segments into 5-second chunks
                    pcg_data, num_chunks, chunk_info = preprocess_pcg(uploaded_pcg)
                    
                    if pcg_data is not None and num_chunks > 0:
                        # Predict using CRNN model (batch prediction on all chunks)
                        pcg_prob, pcg_chunk_probs, pcg_highest_risk_chunk = predict_pcg(
                            pcg_model, pcg_data, num_chunks, chunk_info
                        )
                        
                        if pcg_prob is not None:
                            pcg_signal = pcg_data[0:1]  # Keep first chunk for visualization
                            pcg_chunks_info = chunk_info
                            st.success(f"✅ PCG analysis complete - Analyzed {num_chunks} chunk(s)")
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
            if pcg_signal is not None:
                st.markdown("#### PCG Waveform")
                plot_pcg_waveform(pcg_signal)
                
                st.markdown("#### PCG Spectrogram")
                plot_pcg_spectrogram(pcg_signal, SAMPLE_RATE)
        
        # ====================================================================
        # RESULTS SECTION
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📋 Diagnostic Results")
        
        if ecg_prob is not None and pcg_prob is not None:
            # Both models available: Use fusion
            combined_prob = calculate_combined_risk(ecg_prob, pcg_prob, fusion_method)
            diagnosis, risk_level, color_class = get_diagnosis(combined_prob)
            
            # Display individual model scores
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("ECG Risk Score", f"{ecg_prob*100:.1f}%")
            
            with res_col2:
                pcg_label = f"{pcg_prob*100:.1f}%"
                if pcg_chunks_info and len(pcg_chunks_info) > 1:
                    pcg_label += f" (Max of {len(pcg_chunks_info)} chunks)"
                st.metric("PCG Risk Score", pcg_label)
            
            with res_col3:
                st.metric("Combined Risk", f"{combined_prob*100:.1f}%")
            
            # Display chunk analysis details
            if pcg_chunks_info and pcg_highest_risk_chunk is not None:
                chunk_num, start_time, end_time = pcg_chunks_info[pcg_highest_risk_chunk]
                st.info(f"📊 **PCG Analysis**: Analyzed {len(pcg_chunks_info)} chunk(s). Highest risk detected in chunk #{chunk_num} ({start_time:.1f}s - {end_time:.1f}s)")
            
            # Display final diagnosis
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h3>🎯 Final Multimodal Diagnosis</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Combined Risk Score:</strong> {combined_prob*100:.2f}%</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p><strong>Fusion Method:</strong> {fusion_method.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability comparison chart
            st.markdown("")
            plot_probability_comparison(ecg_prob, pcg_prob, combined_prob)
            
            # Sanity Check Guidance
            st.markdown("---")
            st.markdown("### 🧪 Sanity Check Results")
            
            if ecg_prob < 0.5 and pcg_prob > 0.5:
                st.warning("""
                ⚠️ **"Confused Case" Detected**: Normal ECG but Abnormal PCG
                
                - If result shows GREEN (Normal), the average fusion may be too weak
                - Consider switching to MAX fusion method in the sidebar
                - MAX fusion: `final_risk = max(ECG, PCG)` - more conservative
                """)
            elif ecg_prob > 0.5 and pcg_prob < 0.5:
                st.warning("""
                ⚠️ **"Confused Case" Detected**: Abnormal ECG but Normal PCG
                
                - If result shows GREEN (Normal), the average fusion may be too weak
                - Consider switching to MAX fusion method in the sidebar
                """)
            elif ecg_prob < 0.5 and pcg_prob < 0.5:
                st.success("✅ **Clear Normal Case**: Both modalities indicate low risk")
            elif ecg_prob > 0.5 and pcg_prob > 0.5:
                st.error("⚠️ **Clear Abnormal Case**: Both modalities indicate high risk")
            
        elif ecg_prob is not None:
            # Only ECG available
            diagnosis, risk_level, color_class = get_diagnosis(ecg_prob)
            
            st.metric("ECG Risk Score", f"{ecg_prob*100:.1f}%")
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h3>📊 ECG-Only Diagnosis</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Risk Score:</strong> {ecg_prob*100:.2f}%</p>
                <p><em>Note: PCG data not available for multimodal fusion</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        elif pcg_prob is not None:
            # Only PCG available
            diagnosis, risk_level, color_class = get_diagnosis(pcg_prob)
            
            pcg_display = f"{pcg_prob*100:.1f}%"
            if pcg_chunks_info and len(pcg_chunks_info) > 1:
                pcg_display += f" (Max of {len(pcg_chunks_info)} chunks)"
            st.metric("PCG Risk Score", pcg_display)
            
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <h3>🔊 PCG-Only Diagnosis</h3>
                <h2>{diagnosis}</h2>
                <p><strong>Risk Score:</strong> {pcg_prob*100:.2f}%</p>
                <p><em>Note: ECG data not available for multimodal fusion</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display chunk analysis for PCG-only mode
            if pcg_chunks_info and pcg_highest_risk_chunk is not None:
                chunk_num, start_time, end_time = pcg_chunks_info[pcg_highest_risk_chunk]
                st.info(f"📊 **PCG Analysis**: Analyzed {len(pcg_chunks_info)} chunk(s). Highest risk detected in chunk #{chunk_num} ({start_time:.1f}s - {end_time:.1f}s)")
        else:
            st.error("❌ No valid predictions generated. Please check your input files.")
        
        # ====================================================================
        # TECHNICAL DETAILS (EXPANDABLE)
        # ====================================================================
        with st.expander("🔍 View Technical Details"):
            st.markdown("#### Model Architectures")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **ECG Model (1D-CNN)**
                - Input: (1, 187, 1)
                - Architecture: Convolutional layers
                - Output: Sigmoid probability
                - Trained on: MIT-BIH, PTB Diagnostic ECG
                """)
            
            with col2:
                st.markdown("""
                **PCG Model (CRNN) - Sliding Window**
                - Input per chunk: (1, 5000, 1)
                - Architecture: Conv1D + Bidirectional GRU
                - Preprocessing: Bandpass (20-400Hz), Max-Abs Norm
                - Strategy: Segments audio into 5s chunks
                - Aggregation: Max-Risk (highest probability)
                - Output: Sigmoid probability per chunk
                - Trained on: PhysioNet Heart Sound Challenge
                """)
            
            st.markdown("#### Fusion Strategy")
            if fusion_method == "average":
                st.code("final_risk = (ecg_prob + pcg_prob) / 2.0", language="python")
            else:
                st.code("final_risk = max(ecg_prob, pcg_prob)", language="python")
            
            st.markdown(f"Decision Threshold: **{RISK_THRESHOLD}** (50%)")


if __name__ == "__main__":
    main()
