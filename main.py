"""
FastAPI Server for Multimodal Heart Disease Detection System
Senior AI Engineer & Backend Developer Implementation

Features:
- ECG and PCG signal analysis
- Grad-CAM explainability heatmaps
- Mel-Spectrogram visualization
- Combined risk assessment
"""

import io
import base64
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import tensorflow as tf
from contextlib import asynccontextmanager
from scipy.ndimage import zoom as scipy_zoom


# =============================================================================
# GLOBAL MODEL VARIABLES
# =============================================================================
ecg_model = None
pcg_model = None


# =============================================================================
# APPLICATION LIFESPAN (STARTUP/SHUTDOWN)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown"""
    global ecg_model, pcg_model
    
    print("=" * 60)
    print("Starting Multimodal Heart Disease Detection System...")
    print("=" * 60)
    
    try:
        # Load ECG Model (1D-CNN)
        ecg_model = tf.keras.models.load_model("heart_ecg_model/ecg_model_final.keras")
        print("✓ ECG Model loaded successfully")
        print(f"  - Input shape: {ecg_model.input_shape}")
        print(f"  - Output shape: {ecg_model.output_shape}")
        
        # Load PCG Model (CRNN)
        pcg_model = tf.keras.models.load_model("heart_sound_models/pcg_crnn_model.keras")
        print("✓ PCG Model loaded successfully")
        print(f"  - Input shape: {pcg_model.input_shape}")
        print(f"  - Output shape: {pcg_model.output_shape}")
        
        print("=" * 60)
        print("All models loaded. Server ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down server...")


# =============================================================================
# FASTAPI APPLICATION INITIALIZATION
# =============================================================================
app = FastAPI(
    title="Multimodal Heart Disease Detection API",
    description="ECG and PCG analysis for cardiovascular risk assessment with Grad-CAM explainability",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GRAD-CAM IMPLEMENTATION (CORE EXPLAINABILITY FEATURE)
# =============================================================================
def get_gradcam_heatmap(
    model: tf.keras.Model,
    input_sample: np.ndarray,
    layer_name: str
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a 1D convolutional model.
    
    This function computes class activation maps by:
    1. Building a model that outputs target layer activations + predictions
    2. Computing gradients of the predicted class w.r.t. layer activations
    3. Applying global average pooling to get importance weights
    4. Computing weighted combination of activation maps
    5. Upsampling to match input signal length
    
    Args:
        model: Trained Keras model with Conv1D layers
        input_sample: Input signal with shape (1, length, 1)
        layer_name: Name of the Conv1D layer to visualize
        
    Returns:
        Normalized heatmap array with same length as input signal
    """
    try:
        # Get the target convolutional layer
        target_layer = model.get_layer(layer_name)
        
        # Build gradient model: outputs both layer activation and final prediction
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
        
        # Convert input to tensor
        input_tensor = tf.convert_to_tensor(input_sample, dtype=tf.float32)
        
        # Compute gradients using GradientTape
        with tf.GradientTape() as tape:
            # Watch the input
            tape.watch(input_tensor)
            
            # Forward pass
            layer_output, predictions = grad_model(input_tensor)
            
            # Get the predicted class (for binary: use single output)
            # For regression/binary classification with sigmoid
            predicted_class_score = predictions[0, 0]
        
        # Compute gradients of predicted class w.r.t. layer output
        grads = tape.gradient(predicted_class_score, layer_output)
        
        # Global Average Pooling over the spatial dimension to get weights
        # grads shape: (1, time_steps, filters) -> weights shape: (filters,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Get layer output values
        layer_output_values = layer_output[0]  # Shape: (time_steps, filters)
        
        # Weighted combination: multiply each filter by its importance weight
        # Shape: (time_steps, filters) * (filters,) -> sum over filters -> (time_steps,)
        heatmap = tf.reduce_sum(
            layer_output_values * pooled_grads,
            axis=-1
        )
        
        # Apply ReLU to keep only positive contributions
        heatmap = tf.nn.relu(heatmap)
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Handle edge case of zero heatmap
        if np.max(heatmap) == 0:
            heatmap = np.zeros_like(heatmap)
        else:
            # Normalize to [0, 1]
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Upsample heatmap to match input signal length
        # input_sample shape: (1, length, 1) -> target_length = length
        target_length = input_sample.shape[1]
        current_length = len(heatmap)
        
        if current_length != target_length:
            zoom_factor = target_length / current_length
            heatmap = scipy_zoom(heatmap, zoom_factor, order=1)  # Linear interpolation
            
            # Ensure exact length match
            if len(heatmap) > target_length:
                heatmap = heatmap[:target_length]
            elif len(heatmap) < target_length:
                heatmap = np.pad(heatmap, (0, target_length - len(heatmap)), mode='edge')
        
        # Final normalization after upsampling
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap.astype(np.float32)
        
    except Exception as e:
        print(f"⚠️ Grad-CAM computation failed: {e}")
        # Return zeros as fallback
        target_length = input_sample.shape[1]
        return np.zeros(target_length, dtype=np.float32)


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def preprocess_ecg(file_content: bytes) -> tuple[np.ndarray, List[float]]:
    """
    Preprocess ECG CSV file for model inference.
    
    Processing steps:
    1. Read CSV file (headerless, comma-separated values)
    2. Flatten and convert to float32
    3. Pad/truncate to exactly 187 samples
    4. Min-Max normalize to [0, 1] range
    5. Reshape to (1, 187, 1) for Conv1D input
    
    Args:
        file_content: Raw bytes from uploaded CSV file
        
    Returns:
        Tuple of (preprocessed_array, raw_data_list)
        - preprocessed_array: Shape (1, 187, 1) ready for model
        - raw_data_list: List of 187 normalized float values for frontend
    """
    try:
        # Read CSV file
        df = pd.read_csv(io.BytesIO(file_content), header=None)
        
        # Extract and flatten all values
        ecg_signal = df.values.flatten().astype(np.float32)
        
        # Target length for ECG signal
        target_length = 187
        
        # Pad or truncate to exactly 187 samples
        if len(ecg_signal) < target_length:
            # Pad with zeros if too short
            ecg_signal = np.pad(
                ecg_signal,
                (0, target_length - len(ecg_signal)),
                mode='constant',
                constant_values=0
            )
        elif len(ecg_signal) > target_length:
            # Truncate if too long
            ecg_signal = ecg_signal[:target_length]
        
        # Min-Max normalization to [0, 1]
        min_val = np.min(ecg_signal)
        max_val = np.max(ecg_signal)
        
        if max_val - min_val > 1e-6:
            ecg_signal = (ecg_signal - min_val) / (max_val - min_val)
        else:
            # Handle constant signal
            ecg_signal = np.full_like(ecg_signal, 0.5, dtype=np.float32)
        
        # Create raw data list for frontend (187 points)
        ecg_data = ecg_signal.tolist()
        
        # Reshape for model: (1, 187, 1)
        ecg_array = ecg_signal.reshape(1, 187, 1)
        
        return ecg_array, ecg_data
        
    except Exception as e:
        raise ValueError(f"ECG preprocessing failed: {str(e)}")


def preprocess_pcg(file_content: bytes) -> tuple[np.ndarray, List[float]]:
    """
    Preprocess PCG WAV file for model inference.
    
    Processing steps:
    1. Load WAV using librosa at 1000 Hz sample rate
    2. Pad/truncate to exactly 5 seconds (5000 samples)
    3. Normalize amplitude to [-1, 1] range
    4. Reshape to (1, 5000, 1) for Conv1D input
    
    Args:
        file_content: Raw bytes from uploaded WAV file
        
    Returns:
        Tuple of (preprocessed_array, raw_data_list)
        - preprocessed_array: Shape (1, 5000, 1) ready for model
        - raw_data_list: List of 5000 float values for frontend
    """
    try:
        # PCG Configuration
        SAMPLE_RATE = 1000  # Hz (standard for heart sounds)
        TARGET_LENGTH = 5000  # 5 seconds * 1000 Hz
        
        # Load audio with librosa at specified sample rate
        audio, sr = librosa.load(
            io.BytesIO(file_content),
            sr=SAMPLE_RATE,
            mono=True
        )
        
        # Pad or truncate to exactly 5000 samples (5 seconds)
        if len(audio) < TARGET_LENGTH:
            # Pad with zeros if too short
            audio = np.pad(
                audio,
                (0, TARGET_LENGTH - len(audio)),
                mode='constant',
                constant_values=0
            )
        elif len(audio) > TARGET_LENGTH:
            # Truncate if too long
            audio = audio[:TARGET_LENGTH]
        
        # Normalize to [-1, 1] using max absolute value
        max_abs = np.max(np.abs(audio))
        if max_abs > 1e-6:
            audio = audio / max_abs
        
        # Create raw data list for frontend (5000 points)
        pcg_data = audio.tolist()
        
        # Reshape for model: (1, 5000, 1)
        pcg_array = audio.reshape(1, 5000, 1).astype(np.float32)
        
        return pcg_array, pcg_data
        
    except Exception as e:
        raise ValueError(f"PCG preprocessing failed: {str(e)}")


def generate_spectrogram(audio_data: np.ndarray, sr: int = 1000) -> str:
    """
    Generate Mel-Spectrogram image and convert to Base64 string.
    
    Creates a visual representation of the frequency content over time,
    which is useful for identifying heart sound patterns (S1, S2, murmurs).
    
    Args:
        audio_data: Audio signal array
        sr: Sample rate (default: 1000 Hz)
        
    Returns:
        Base64 encoded PNG image string with data URI prefix
    """
    try:
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Generate Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_mels=128,
            fmax=500,  # Heart sounds are below 500 Hz
            hop_length=64
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Display spectrogram
        img = librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='magma',
            hop_length=64
        )
        
        # Add labels and colorbar
        ax.set_title('Mel-Spectrogram', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        fig.colorbar(img, ax=ax, format='%+2.0f dB', shrink=0.8)
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to Base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        # Return with data URI prefix
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"⚠️ Spectrogram generation failed: {e}")
        return ""


# =============================================================================
# HELPER FUNCTION TO FIND LAST CONV1D LAYER
# =============================================================================
def find_last_conv1d_layer(model: tf.keras.Model) -> str:
    """
    Find the name of the last Conv1D layer in the model.
    
    Args:
        model: Keras model
        
    Returns:
        Name of the last Conv1D layer, or None if not found
    """
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv_layer = layer.name
    return last_conv_layer


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Multimodal Heart Disease Detection API",
        "version": "2.0.0",
        "models_loaded": ecg_model is not None and pcg_model is not None
    }


@app.post("/predict")
async def predict(
    ecg_file: Optional[UploadFile] = File(None),
    pcg_file: Optional[UploadFile] = File(None)
):
    """
    Main prediction endpoint for ECG and/or PCG analysis.
    
    Accepts optional ECG (CSV) and PCG (WAV) files, performs predictions,
    generates Grad-CAM heatmaps for explainability, and returns comprehensive
    analysis results.
    
    Args:
        ecg_file: Optional CSV file containing ECG signal (187 samples)
        pcg_file: Optional WAV file containing heart sound (5 seconds)
        
    Returns:
        JSON response with:
        - ecg_risk: Float risk score from ECG model (0-1)
        - pcg_risk: Float risk score from PCG model (0-1)
        - combined_risk: Average of ECG and PCG risks
        - ecg_data: Raw ECG signal (187 points)
        - ecg_heatmap: Grad-CAM heatmap for ECG (187 points)
        - pcg_data: Raw PCG signal (5000 points)
        - pcg_heatmap: Grad-CAM heatmap for PCG (5000 points)
        - spectrogram_image: Base64 encoded Mel-Spectrogram
    """
    
    # Validation: At least one file must be provided
    if ecg_file is None and pcg_file is None:
        raise HTTPException(
            status_code=400,
            detail="At least one file (ECG or PCG) must be provided"
        )
    
    # Initialize response structure (matches required JSON format)
    response = {
        "ecg_risk": None,
        "pcg_risk": None,
        "combined_risk": None,
        "ecg_data": [],
        "ecg_heatmap": [],
        "pcg_data": [],
        "pcg_heatmap": [],
        "spectrogram_image": ""
    }
    
    try:
        # =====================================================================
        # PROCESS ECG FILE (if provided)
        # =====================================================================
        if ecg_file is not None:
            print(f"Processing ECG file: {ecg_file.filename}")
            
            # Read and preprocess ECG
            ecg_content = await ecg_file.read()
            ecg_array, ecg_data = preprocess_ecg(ecg_content)
            
            # Run ECG model prediction
            ecg_prediction = ecg_model.predict(ecg_array, verbose=0)
            ecg_risk = float(ecg_prediction[0][0])
            
            # Generate Grad-CAM heatmap for ECG
            # TODO: Change "conv1d_last_ecg" to the actual last Conv1D layer name in your ECG model
            # Use find_last_conv1d_layer(ecg_model) to discover it, or check model.summary()
            ecg_layer_name = "conv1d_last_ecg"  # CHANGE THIS to your actual layer name
            
            # Try to auto-detect the last Conv1D layer
            detected_layer = find_last_conv1d_layer(ecg_model)
            if detected_layer:
                ecg_layer_name = detected_layer
                print(f"  Using detected ECG Conv1D layer: {ecg_layer_name}")
            
            ecg_heatmap = get_gradcam_heatmap(ecg_model, ecg_array, ecg_layer_name)
            
            # Populate response
            response["ecg_risk"] = ecg_risk
            response["ecg_data"] = ecg_data
            response["ecg_heatmap"] = ecg_heatmap.tolist()
            
            print(f"  ECG Risk Score: {ecg_risk:.4f}")
        
        # =====================================================================
        # PROCESS PCG FILE (if provided)
        # =====================================================================
        if pcg_file is not None:
            print(f"Processing PCG file: {pcg_file.filename}")
            
            # Read and preprocess PCG
            pcg_content = await pcg_file.read()
            pcg_array, pcg_data = preprocess_pcg(pcg_content)
            
            # Run PCG model prediction
            pcg_prediction = pcg_model.predict(pcg_array, verbose=0)
            pcg_risk = float(pcg_prediction[0][0])
            
            # Generate spectrogram from raw audio data
            spectrogram_image = generate_spectrogram(np.array(pcg_data))
            
            # Generate Grad-CAM heatmap for PCG
            # TODO: Change "conv1d_last_pcg" to the actual last Conv1D layer name in your PCG model
            # Use find_last_conv1d_layer(pcg_model) to discover it, or check model.summary()
            pcg_layer_name = "conv1d_last_pcg"  # CHANGE THIS to your actual layer name
            
            # Try to auto-detect the last Conv1D layer
            detected_layer = find_last_conv1d_layer(pcg_model)
            if detected_layer:
                pcg_layer_name = detected_layer
                print(f"  Using detected PCG Conv1D layer: {pcg_layer_name}")
            
            pcg_heatmap = get_gradcam_heatmap(pcg_model, pcg_array, pcg_layer_name)
            
            # Populate response
            response["pcg_risk"] = pcg_risk
            response["pcg_data"] = pcg_data
            response["pcg_heatmap"] = pcg_heatmap.tolist()
            response["spectrogram_image"] = spectrogram_image
            
            print(f"  PCG Risk Score: {pcg_risk:.4f}")
        
        # =====================================================================
        # CALCULATE COMBINED RISK
        # =====================================================================
        if response["ecg_risk"] is not None and response["pcg_risk"] is not None:
            combined_risk = (response["ecg_risk"] + response["pcg_risk"]) / 2.0
            response["combined_risk"] = combined_risk
            print(f"  Combined Risk Score: {combined_risk:.4f}")
        
        print("✓ Prediction completed successfully")
        return JSONResponse(content=response)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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


@app.get("/model-info")
async def model_info():
    """Get information about loaded models including layer names for Grad-CAM"""
    info = {
        "ecg_model": None,
        "pcg_model": None
    }
    
    if ecg_model is not None:
        ecg_layers = [layer.name for layer in ecg_model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        info["ecg_model"] = {
            "input_shape": str(ecg_model.input_shape),
            "output_shape": str(ecg_model.output_shape),
            "conv1d_layers": ecg_layers,
            "last_conv1d_layer": ecg_layers[-1] if ecg_layers else None
        }
    
    if pcg_model is not None:
        pcg_layers = [layer.name for layer in pcg_model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        info["pcg_model"] = {
            "input_shape": str(pcg_model.input_shape),
            "output_shape": str(pcg_model.output_shape),
            "conv1d_layers": pcg_layers,
            "last_conv1d_layer": pcg_layers[-1] if pcg_layers else None
        }
    
    return info


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
