import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom


def build_grad_model(model, input_shape, layer_name):
    """
    Builds a gradient model using Functional Graph Reconstruction.
    
    This approach avoids the "layer has never been called" error by reconstructing
    the model graph from scratch instead of relying on model.inputs.
    
    Args:
        model (tf.keras.Model): The original Keras model
        input_shape (tuple): Shape of the input (without batch dimension), e.g., (187, 1)
        layer_name (str): Name of the target convolutional layer
        
    Returns:
        tf.keras.Model: A new model that outputs [target_layer_output, final_output]
        
    Raises:
        ValueError: If the layer_name is not found in the model
    """
    # Verify the layer exists
    target_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break
    
    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Create a new input tensor (disconnected from the original model graph)
    new_input = tf.keras.Input(shape=input_shape)
    
    # Re-trace through all model layers
    x = new_input
    target_output = None
    
    for layer in model.layers:
        x = layer(x)
        
        # Capture the output of the target layer
        if layer.name == layer_name:
            target_output = x
    
    # Final output is the last layer's output
    final_output = x
    
    # Build the gradient model
    grad_model = tf.keras.models.Model(inputs=new_input, outputs=[target_output, final_output])
    
    return grad_model


def get_gradcam_heatmap(model, input_array, layer_name):
    """
    Generates a Grad-CAM heatmap for a given Keras model and input using
    Functional Graph Reconstruction to avoid "layer has never been called" errors.

    Args:
        model (tf.keras.Model): The Keras model to analyze.
        input_array (np.ndarray): The input data for which to generate the heatmap.
                                  Should be preprocessed and have the batch dimension.
                                  e.g., (1, 187, 1) for an ECG signal.
        layer_name (str): The name of the convolutional layer to visualize.

    Returns:
        list: A 1D list of float values representing the normalized heatmap,
              resized to match the input signal length. Returns a list of zeros
              if the layer_name is invalid or if any error occurs.
    """
    input_signal_length = input_array.shape[1]
    
    try:
        # Extract input shape (without batch dimension)
        input_shape = input_array.shape[1:]  # e.g., (187, 1) or (5000, 1)
        
        # Build the gradient model using Functional Graph Reconstruction
        grad_model = build_grad_model(model, input_shape, layer_name)
        
    except ValueError as e:
        print(f"❌ Error: Invalid layer name '{layer_name}'.")
        print("Available layer names:")
        for layer in model.layers:
            print(f"  - {layer.name}")
        
        # Return a zero array with the same length as the input signal
        return [0.0] * input_signal_length
        
    except Exception as e:
        print(f"❌ Error building gradient model: {e}")
        print(f"   Model: {model.name}, Layer: {layer_name}, Input shape: {input_array.shape}")
        return [0.0] * input_signal_length

    try:
        # Convert input to tensor if needed
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Pass the input through the grad_model to get feature maps and predictions
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            
            # Get the score for the top predicted class
            top_pred_index = tf.argmax(predictions[0])
            top_class_channel = predictions[:, top_pred_index]

        # Compute the gradients of the top class score with respect to the feature maps
        grads = tape.gradient(top_class_channel, conv_outputs)
        
        # Handle case where gradients are None
        if grads is None:
            print(f"⚠️ Warning: Gradients are None for layer '{layer_name}'")
            return [0.0] * input_signal_length

        # Global average pooling of the gradients to get the weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # Get the feature maps from the convolutional layer output
        conv_outputs = conv_outputs[0]
        
        # Multiply the feature maps by the weights
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Convert to numpy for processing
        heatmap = heatmap.numpy()

        # Upsample the heatmap to match the original input size
        # The heatmap is 1D, so we get its size
        current_length = len(heatmap) if heatmap.ndim > 0 else 0
        
        # Use scipy's zoom for smooth upsampling
        if current_length > 0 and current_length != input_signal_length:
            zoom_factor = input_signal_length / current_length
            heatmap = zoom(heatmap, zoom_factor)
        elif current_length == 0:
            heatmap = np.zeros(input_signal_length)

        # Normalize the heatmap to the 0-1 range
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.tolist()
        
    except Exception as e:
        print(f"❌ Error computing Grad-CAM heatmap: {e}")
        print(f"   Model: {model.name}, Layer: {layer_name}, Input shape: {input_array.shape}")
        import traceback
        traceback.print_exc()
        # Return a zero array as fallback to prevent server crash
        return [0.0] * input_signal_length
