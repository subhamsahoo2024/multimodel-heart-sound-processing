"""
FastAPI Server for Multimodal Heart Disease Detection System
Senior Backend Engineer Implementation
"""

import io
import base64
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import tensorflow as tf
from contextlib import asynccontextmanager
from scipy.signal import butter, filtfilt
from explainability import get_gradcam_heatmap


# Global model variables
ecg_model = None
pcg_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown with warmup routine"""
    global ecg_model, pcg_model
    
    print("Loading models...")
    try:
        # Load ECG Model (1D-CNN)
        ecg_model = tf.keras.models.load_model("heart_ecg_model/ecg_model_final.keras")
        print("✓ ECG Model loaded successfully")
        
        # Load PCG Model (CRNN)
        pcg_model = tf.keras.models.load_model("heart_sound_models/pcg_crnn_model.keras")
        print("✓ PCG Model loaded successfully")
        
        # CRITICAL: Warmup routine to build computation graphs
        print("Warming up models with dummy predictions...")
        
        # Warmup ECG model with dummy input (1, 187, 1)
        dummy_ecg = np.zeros((1, 187, 1), dtype=np.float32)
        _ = ecg_model.predict(dummy_ecg, verbose=0)
        print("✓ ECG Model warmed up successfully")
        
        # Warmup PCG model with dummy input (1, 5000, 1)
        dummy_pcg = np.zeros((1, 5000, 1), dtype=np.float32)
        _ = pcg_model.predict(dummy_pcg, verbose=0)
        print("✓ PCG Model warmed up successfully")
        
        print("✓ All models ready for inference and explainability!")
        
    except Exception as e:
        print(f"❌ Error loading/warming up models: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Heart Disease Detection API",
    description="ECG and PCG analysis for cardiovascular risk assessment",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_ecg(file_content: bytes) -> tuple[np.ndarray, List[List[float]]]:
    """
    Preprocess ECG CSV file according to research standards.
    
    Args:
        file_content: Raw bytes from uploaded CSV file
        
    Returns:
        Tuple of (preprocessed_array, plot_data)
        - preprocessed_array: Shape (1, 187, 1) ready for model
        - plot_data: List of [x, y] coordinates for frontend plotting
    """
    try:
        # Read CSV file
        df = pd.read_csv(io.BytesIO(file_content), header=None)
        
        # Extract and flatten all values
        ecg_signal = df.values.flatten().astype(np.float32)
        
        # Pad or truncate to exactly 187 samples
        target_length = 187
        if len(ecg_signal) < target_length:
            # Pad with zeros if too short
            ecg_signal = np.pad(ecg_signal, (0, target_length - len(ecg_signal)), mode='constant', constant_values=0)
        elif len(ecg_signal) > target_length:
            # Take first 187 samples
            ecg_signal = ecg_signal[:target_length]
        
        # Normalize using Min-Max scaling (0-1) - matches app3.py
        min_val = np.min(ecg_signal)
        max_val = np.max(ecg_signal)
        
        if max_val - min_val > 1e-6:  # Avoid division by near-zero
            ecg_signal = (ecg_signal - min_val) / (max_val - min_val)
        else:
            # If constant signal, set to 0.5 (middle of 0-1 range)
            ecg_signal = np.full_like(ecg_signal, 0.5, dtype=np.float32)
        
        # Prepare plot data for frontend
        plot_data = [[int(i), float(val)] for i, val in enumerate(ecg_signal)]
        
        # Reshape for model: (1, 187, 1)
        ecg_array = ecg_signal.reshape(1, 187, 1)
        
        return ecg_array, plot_data
        
    except Exception as e:
        raise ValueError(f"ECG preprocessing failed: {str(e)}")


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to remove noise"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_pcg(file_content: bytes) -> tuple[np.ndarray, List[float], str]:
    """
    Preprocess PCG audio file using sliding window technique with raw waveform.
    
    Args:
        file_content: Raw bytes from uploaded WAV file
        
    Returns:
        Tuple of (preprocessed_chunks, waveform_data, spectrogram_base64)
        - preprocessed_chunks: Array of shape (N, 5000, 1) for model (raw waveform)
        - waveform_data: Downsampled amplitude values for frontend
        - spectrogram_base64: Base64 encoded spectrogram image
    """
    try:
        # PCG Configuration (Must match training)
        SAMPLE_RATE = 1000  # Hz
        TARGET_LENGTH = 5000  # 5 seconds * 1000 Hz
        LOWCUT = 20  # Hz
        HIGHCUT = 400  # Hz
        
        # Load audio with librosa at 1000 Hz sample rate
        audio, sr = librosa.load(io.BytesIO(file_content), sr=SAMPLE_RATE, mono=True)
        
        # Apply bandpass filter (20-400 Hz) to entire audio
        audio = butter_bandpass_filter(audio, LOWCUT, HIGHCUT, SAMPLE_RATE)
        
        # Calculate number of 5-second chunks
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / TARGET_LENGTH))
        
        # Segment and preprocess each chunk
        preprocessed_chunks = []
        
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
        
        # Stack all chunks into batch: (num_chunks, 5000, 1)
        batch_data = np.vstack(preprocessed_chunks)
        
        # Prepare waveform data for frontend (downsample to ~1000 points)
        downsample_factor = max(1, len(audio) // 1000)
        waveform_data = audio[::downsample_factor].tolist()
        
        # Generate spectrogram image for frontend
        spectrogram_base64 = generate_spectrogram_image(audio, SAMPLE_RATE)
        
        return batch_data, waveform_data, spectrogram_base64
        
    except Exception as e:
        raise ValueError(f"PCG preprocessing failed: {str(e)}")


def generate_spectrogram_image(audio: np.ndarray, sr: int) -> str:
    """
    Generate a Mel-Spectrogram image and convert to Base64.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Base64 encoded PNG image string
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=500)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        img = librosa.display.specshow(
            mel_spec_db, 
            sr=sr, 
            x_axis='time', 
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        ax.set_title('Mel-Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Convert to Base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"Spectrogram generation warning: {e}")
        return ""


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Multimodal Heart Disease Detection API",
        "models_loaded": ecg_model is not None and pcg_model is not None
    }


@app.post("/predict")
async def predict(
    ecg_file: Optional[UploadFile] = File(None),
    pcg_file: Optional[UploadFile] = File(None)
):
    """
    Main prediction endpoint for ECG and/or PCG analysis.
    
    Args:
        ecg_file: Optional CSV file containing ECG signal
        pcg_file: Optional WAV file containing heart sound
        
    Returns:
        JSON response with risk scores and visualization data
    """
    
    # Validation: At least one file must be provided
    if ecg_file is None and pcg_file is None:
        raise HTTPException(
            status_code=400, 
            detail="At least one file (ECG or PCG) must be provided"
        )
    
    # Initialize response structure
    response = {
        "ecg_risk": None,
        "pcg_risk": None,
        "combined_risk": None,
        "ecg_plot_data": [],
        "pcg_waveform_data": [],
        "pcg_spectrogram": "",
        "ecg_heatmap": [],
        "pcg_heatmap": []
    }
    
    try:
        # Process ECG if provided
        if ecg_file is not None:
            print(f"Processing ECG file: {ecg_file.filename}")
            ecg_content = await ecg_file.read()
            ecg_array, ecg_plot_data = preprocess_ecg(ecg_content)
            
            # Run ECG model prediction
            ecg_prediction = ecg_model.predict(ecg_array, verbose=0)
            ecg_risk = float(ecg_prediction[0][0])

            # Generate Grad-CAM heatmap for ECG
            # The layer name 'conv1d_3' should be verified from your model summary
            ecg_heatmap = get_gradcam_heatmap(ecg_model, ecg_array, 'conv1d_2')
            
            response["ecg_risk"] = ecg_risk
            response["ecg_plot_data"] = ecg_plot_data
            response["ecg_heatmap"] = ecg_heatmap
            print(f"ECG Risk Score: {ecg_risk:.4f}")
        
        # Process PCG if provided
        if pcg_file is not None:
            print(f"Processing PCG file: {pcg_file.filename}")
            pcg_content = await pcg_file.read()
            batch_data, waveform_data, spectrogram_base64 = preprocess_pcg(pcg_content)
            
            # Run PCG model prediction on all chunks
            pcg_predictions = pcg_model.predict(batch_data, verbose=0)
            
            # Take maximum risk across all chunks (worst-case scenario)
            pcg_risk = float(np.max(pcg_predictions))

            # Generate and stitch Grad-CAM heatmaps for PCG
            stitched_heatmap = []
            # The layer name 'conv1d_5' should be verified from your model summary
            for i in range(batch_data.shape[0]):
                chunk = np.expand_dims(batch_data[i], axis=0) # Shape (1, 5000, 1)
                heatmap = get_gradcam_heatmap(pcg_model, chunk, 'conv2')
                stitched_heatmap.extend(heatmap)
            
            # Downsample the stitched heatmap to match the waveform data size for visualization
            pcg_heatmap = stitched_heatmap[::10]
            
            response["pcg_risk"] = pcg_risk
            response["pcg_waveform_data"] = waveform_data
            response["pcg_spectrogram"] = spectrogram_base64
            response["pcg_heatmap"] = pcg_heatmap
            print(f"PCG Risk Score: {pcg_risk:.4f} (max of {len(pcg_predictions)} chunks)")
        
        # Calculate combined risk if both provided
        if response["ecg_risk"] is not None and response["pcg_risk"] is not None:
            combined_risk = (response["ecg_risk"] + response["pcg_risk"]) / 2.0
            response["combined_risk"] = combined_risk
            print(f"Combined Risk Score: {combined_risk:.4f}")
        
        return JSONResponse(content=response)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "ecg_model": "loaded" if ecg_model is not None else "not loaded",
            "pcg_model": "loaded" if pcg_model is not None else "not loaded"
        },
        "endpoints": ["/", "/predict", "/health"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
