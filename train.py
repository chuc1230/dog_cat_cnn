import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 1. Chuẩn bị dữ liệu
train_dir = "data/train"
val_dir = "data/validation"

# ImageDataGenerator dùng để chuẩn hóa ảnh và áp dụng augment
train_datagen = ImageDataGenerator(
    rescale=1./255,                
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Chỉ chuẩn hóa ảnh validation, không augment 
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary" # vì là phân loại nhị phân (chó và mèo), nên dùng binary để gán nhãn 
)
# Lưu ý quan trọng: thứ tự mã hóa lớp (0 hay 1) do Keras quyết định theo tên folder (sắp xếp chữ cái). 
# Ở đây, folder "cats" đứng trước "dogs" nên mèo sẽ được mã hóa là 0, chó là 1.
# Có thể kiểm tra bằng lệnh: print(train_generator.class_indices)
# Nhãn này quan trọng, cần được kiểm soát để tránh nhầm lẫn khi dự đoán.
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# 2. Xây dựng mô hình CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)), # tích chập 32 bộ lọc 3x3
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"), # tích chập 64 bộ lọc 3x3
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"), # tích chập 128 bộ lọc 3x3
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), # chuyển từ ma trận 2D sang vector 1D
    tf.keras.layers.Dense(512, activation="relu"), # lớp fully connected với 512 nút để trích xuất đặc trưng
    tf.keras.layers.Dense(1, activation="sigmoid")# lớp đầu ra với 1 nút (vì phân loại nhị phân) dùng hàm sigmoid
])

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 3. Train mô hình
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# 4. Vẽ kết quả accuracy
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.show()

# 5. Lưu model
os.makedirs("models", exist_ok=True)
model.save("models/dog_cat_cnn.h5")
print(" Model đã được lưu vào models/dog_cat_cnn.h5")
