# dog_cat_cnn
- Project nhận diện chó, mèo sử dụng CNN (https://topdev.vn/blog/thuat-toan-cnn-convolutional-neural-network/)
- Data được lấy từ Kaggle
- Ngôn ngữ: python 
- Thư viện sử dụng: TensorFlow, NumPy, SciPy
  'pip install tensorflow',
  'pip install numpy',
  'pip install scipy',
  + TensorFlow: xây dựng và huấn luyện CNN
  + Numpy, Scipy: xử lý dữ liệu
- Chạy chương trình: py train.py
 -> Model sau khi train sẽ được lưu tại  /models/dog_cat_cnn.h5 , model chỉ lưu 1 bản mỗi khi chạy file train.py
  -> Chạy file test.py để nhậ diện hình ảnh chó, mèo( thay đổi đường dẫn hình ảnh trước khi chạy) 
