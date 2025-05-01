# train_model.py

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import os

# 1. Define labeled data (extend as needed)
data = [
    ("Create a sales order for customer X with products A and B", "create_sales_order"),
    ("I want to place a sales order for customer Y with products C and D", "create_sales_order"),
    ("Create a purchase order with supplier Z for product E", "create_purchase_order"),
    ("Place a purchase order with supplier W for products F and G", "create_purchase_order"),
    ("How many units of FRISOLAC are in stock?", "check_stock"),
    ("Please create a new purchase order", "create_purchase_order"),
    ("Order items from supplier M with quantity 5 of product J", "create_purchase_order"),
    ("Can you help me create a sales order for customer L?", "create_sales_order"),
    ("Sales order for customer A with products T and U", "create_sales_order"),
    ("Purchase order with supplier B for products V, W", "create_purchase_order"),
    ("Order product M for customer O", "create_sales_order"),
    ("Place a purchase order for 10 units of product P with supplier Q", "create_purchase_order"),
    ("I need to create a sales order for customer J with products H, I", "create_sales_order"),
    ("I want to check stock availability for product ABC", "check_stock"),
    ("How much inventory do we have for product XYZ?", "check_stock"),
    ("Generate a purchase order for 20 chairs for supplier DEF", "create_purchase_order"),
    ("Create a sales order for 5 items of product GHI", "create_sales_order"),
    ("Do we still have FRISOLAC in stock?", "check_stock"),
    ("How many units are left of product A?", "check_stock"),
    ("Check inventory for product ABC", "check_stock"),
    ("I want to see current stock for product B", "check_stock"),
    ("What is the available quantity of FRISO?", "check_stock"),
    ("Is product X still in stock?", "check_stock"),
    ("Show me inventory level for 'baby milk'", "check_stock"),
    ("Get me stock status of Glico ICREO", "check_stock"),
    ("Check how much of product X is in warehouse", "check_stock"),
    ("How much stock do we have for each product?", "check_stock"),
    ("Create a sales order for customer ABC with 5 boxes of FRISO", "create_sales_order"),
    ("I want to place a sales order", "create_sales_order"),
    ("Can you help me create an SO?", "create_sales_order"),
    ("Make a sales order for customer Peter", "create_sales_order"),
    ("Order 10 units of FRISO for client X", "create_sales_order"),
    ("Add a new sale for Glico product", "create_sales_order"),
    ("Sales order: customer Y needs 2 cases of milk", "create_sales_order"),
    ("Register new sales order for 'PLUS ONE'", "create_sales_order"),
    ("Place an order to sell 15 items of ABC", "create_sales_order"),
    ("Generate a sales invoice for FRISOLAC", "create_sales_order"),
    ("Create a purchase order from supplier M&B", "create_purchase_order"),
    ("Order 100 bottles of hand sanitizer from supplier XYZ", "create_purchase_order"),
    ("Place a PO for 20 units of product DEF", "create_purchase_order"),
    ("I want to buy stock for Glico milk", "create_purchase_order"),
    ("Generate PO for 30 units from ABC supplier", "create_purchase_order"),
    ("New purchase order: 10 cartons of FRISO", "create_purchase_order"),
    ("I need to restock product P by ordering from vendor V", "create_purchase_order"),
    ("Send PO to supplier for 100 face masks", "create_purchase_order"),
    ("Check the status of sales order S00032", "find_by_number"),
    ("What is the status of PO00018?", "find_by_number"),
    ("Find sales order S00045 for me", "find_by_number"),

    ("Show me all orders for customer X", "find_by_partner"),
    ("List purchase orders for supplier ABC", "find_by_partner"),

    ("List my last 5 sales orders", "list_recent"),
    ("Show me the 10 most recent purchase orders", "list_recent"),
    ("Give me the last 3 orders", "list_recent"),
]

# Split data
X = [text for text, label in data]
y = [label for text, label in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with combined word+char n-grams and a stacking classifier
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('word_ngrams', TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_df=0.9, min_df=1)),
        ('char_ngrams', TfidfVectorizer(ngram_range=(2, 5), analyzer='char', max_df=0.9, min_df=1)),
    ])),
    ('classifier', StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('svc', LinearSVC()),
            ('rf', RandomForestClassifier(n_estimators=100)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    ))
])

# Optionally tune hyperparameters
param_grid = {
    'features__word_ngrams__ngram_range': [(1, 1), (1, 2)],
    'classifier__rf__n_estimators': [50, 100],
    'classifier__lr__C': [0.1, 1, 10],
}
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Persist the trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_dir, '../../static/data/intent_classifier_advanced.pkl')

# Tạo thư mục nếu chưa có
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Ghi file
joblib.dump(best_model, save_path)
