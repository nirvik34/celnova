# This script must be run in a Python environment with Keras 3.x and TensorFlow 2.15+ installed.
# It loads the legacy .h5 model and saves it in the new .keras format, compatible with Keras 3.x.

from keras.models import load_model

# Path to your legacy model
legacy_model_path = 'model/model.h5'
# Path to save the new Keras 3.x model
new_model_path = 'model/model_keras3.keras'

# Load the legacy model
model = load_model(legacy_model_path, compile=False)

# Save in the new Keras 3.x format
model.save(new_model_path)

print(f"Model successfully resaved in Keras 3.x format: {new_model_path}")
