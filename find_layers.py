"""
Diagnostic Script: Find Conv1D Layer Names for Grad-CAM
========================================================
This script loads both ECG and PCG Keras models and identifies
the last Conv1D layer in each, which is required for Grad-CAM.

Usage:
    python find_layers.py
    
    OR with uv:
    uv run python find_layers.py
"""

import tensorflow as tf
import os

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def find_conv1d_layers(model, model_name: str) -> dict:
    """
    Find all Conv1D layers in a model and identify the last one.
    
    Args:
        model: Loaded Keras model
        model_name: Name for display purposes
        
    Returns:
        Dictionary with layer information
    """
    conv1d_layers = []
    all_layers = []
    
    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__
        }
        all_layers.append(layer_info)
        
        # Check if layer name contains 'conv1d' (case-insensitive)
        if 'conv1d' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv1D):
            conv1d_layers.append(layer.name)
    
    return {
        "model_name": model_name,
        "total_layers": len(all_layers),
        "conv1d_layers": conv1d_layers,
        "last_conv1d": conv1d_layers[-1] if conv1d_layers else None,
        "all_layers": all_layers
    }


def print_model_summary(info: dict, verbose: bool = False):
    """Print formatted model layer information."""
    print(f"\n{'='*60}")
    print(f"Model: {info['model_name']}")
    print(f"{'='*60}")
    print(f"Total Layers: {info['total_layers']}")
    print(f"Conv1D Layers Found: {len(info['conv1d_layers'])}")
    
    if info['conv1d_layers']:
        print(f"\nAll Conv1D Layers:")
        for i, layer_name in enumerate(info['conv1d_layers'], 1):
            marker = " <-- LAST (use this for Grad-CAM)" if layer_name == info['last_conv1d'] else ""
            print(f"  {i}. {layer_name}{marker}")
    else:
        print("\n⚠️  No Conv1D layers found in this model!")
    
    if verbose:
        print(f"\nAll Layers:")
        for layer in info['all_layers']:
            print(f"  - {layer['name']} ({layer['type']})")


def main():
    print("\n" + "="*60)
    print("    GRAD-CAM LAYER FINDER FOR HEART DISEASE DETECTION")
    print("="*60)
    
    # Model paths (relative to backend directory)
    ecg_model_path = "heart_ecg_model/ecg_model_final.keras"
    pcg_model_path = "heart_sound_models/pcg_crnn_model.keras"
    
    results = {}
    
    # Load and analyze ECG Model
    print("\n📂 Loading ECG Model...")
    try:
        ecg_model = tf.keras.models.load_model(ecg_model_path)
        print("✓ ECG Model loaded successfully")
        results['ecg'] = find_conv1d_layers(ecg_model, "ECG Model")
        print_model_summary(results['ecg'])
    except Exception as e:
        print(f"❌ Failed to load ECG Model: {e}")
        results['ecg'] = None
    
    # Load and analyze PCG Model
    print("\n📂 Loading PCG Model...")
    try:
        pcg_model = tf.keras.models.load_model(pcg_model_path)
        print("✓ PCG Model loaded successfully")
        results['pcg'] = find_conv1d_layers(pcg_model, "PCG Model")
        print_model_summary(results['pcg'])
    except Exception as e:
        print(f"❌ Failed to load PCG Model: {e}")
        results['pcg'] = None
    
    # Print final summary for easy copy-paste
    print("\n" + "="*60)
    print("    SUMMARY - COPY THESE VALUES TO main.py")
    print("="*60)
    
    ecg_last = results['ecg']['last_conv1d'] if results['ecg'] else "NOT_FOUND"
    pcg_last = results['pcg']['last_conv1d'] if results['pcg'] else "NOT_FOUND"
    
    print(f"\n  ECG Last Conv1D Layer: \"{ecg_last}\"")
    print(f"  PCG Last Conv1D Layer: \"{pcg_last}\"")
    
    print("\n" + "-"*60)
    print("  UPDATE main.py WITH THESE VALUES:")
    print("-"*60)
    print(f"""
  # In the /predict endpoint, find these lines and update:
  
  # For ECG (around line 390):
  ecg_layer_name = "{ecg_last}"
  
  # For PCG (around line 420):
  pcg_layer_name = "{pcg_last}"
""")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
