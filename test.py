import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model đã train
model = tf.keras.models.load_model("models/dog_cat_cnn.h5")

# Dự đoán ảnh mới
img_path = "data/test/cats/cat.4004.jpg"  # đổi đường dẫn sang ảnh muốn test
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print(" Đây là chó ")
else:
    print(" Đây là mèo ")
