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
- Chạy chương trình:
  1. py train.py
     <img width="1296" height="464" alt="image" src="https://github.com/user-attachments/assets/5b2aeb7a-2874-4b62-a0bf-ccdfc424bcd0" />
     <img width="643" height="556" alt="image" src="https://github.com/user-attachments/assets/1251903f-90b1-440b-9c7c-545999bee1ca" />

     + Các chỉ số :
       * accuracy → độ chính xác trên tập huấn luyện (train set).
       * loss → giá trị hàm mất mát (binary crossentropy). Thấp hơn nghĩa là mô hình dự đoán tốt hơn.
       * val_accuracy → độ chính xác trên tập validation (kiểm chứng). Kiểm tra xem mô hình có "học vẹt" hay không.
       

  3. Model sau khi train sẽ được lưu tại  /models/dog_cat_cnn.h5 , model chỉ lưu 1 bản mỗi khi chạy file train.py
  4. Chạy file test.py để nhậ diện hình ảnh chó, mèo( thay đổi đường dẫn hình ảnh trước khi chạy) 
