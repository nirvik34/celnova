import tensorflow as tf

# Load the model using tf.keras (TensorFlow 2.x)
model = tf.keras.models.load_model('model/model.h5')

# Save the model in the new Keras format
model.save('model/model.keras', save_format='keras')

print('Model has been resaved in Keras v3 format as model/model.keras')
