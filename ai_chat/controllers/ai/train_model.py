# train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Dữ liệu huấn luyện mẫu
data = [
    ("Create a sales order for customer X with products A and B", "Sales Order"),
    ("I want to place a sales order for customer Y with products C and D", "Sales Order"),
    ("Create a purchase order with supplier Z for product E", "Purchase Order"),
    ("Place a purchase order with supplier W for products F and G", "Purchase Order"),
    ("I need to create a sales order for customer J with products H, I", "Sales Order"),
    ("Order items from supplier M with quantity 5 of product J", "Purchase Order"),
    ("Can you help me create a sales order for customer L?", "Sales Order"),
    ("Create a purchase order from supplier N with products K and L", "Purchase Order"),
    ("Order product M for customer O", "Sales Order"),
    ("Place a purchase order for 10 units of product P with supplier Q", "Purchase Order"),
    ("Sales order for customer A with products T and U", "Sales Order"),
    ("Purchase order with supplier B for products V, W", "Purchase Order")
]

# Tách dữ liệu thành câu hỏi (X) và nhãn (y)
X = [item[0] for item in data]
y = [item[1] for item in data]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình SVM với pipeline TfidfVectorizer và SVC
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Lưu mô hình đã huấn luyện
import joblib
joblib.dump(model, 'ai_bot/static/data/sales_purchase_classifier.pkl')
