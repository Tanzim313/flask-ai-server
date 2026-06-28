import tensorflow as tf
import os

model_path = os.path.join("models", "plant_disease_final_model.h5")
if not os.path.exists(model_path):
    print("Model not found at", model_path)
    exit(1)

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("=== MODEL SUMMARY / INFO ===")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("\n=== LAYERS ===")
    for i, layer in enumerate(model.layers):
        # Print class name and basic config if available
        config = layer.get_config()
        # print first few layers config
        if i < 15 or i >= len(model.layers) - 5:
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
            if "scale" in config or "offset" in config:
                print(f"  -> scale/offset: {config.get('scale')}, {config.get('offset')}")
            # If it's a functional/sequential model, print its name
            if isinstance(layer, tf.keras.Model):
                print(f"  -> Inner Model Input: {layer.input_shape}, Output: {layer.output_shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
