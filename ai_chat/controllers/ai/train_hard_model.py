# train_model.py

import numpy as np
import joblib
import re
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Define labeled data (extend as needed)
data = [
    ("Create a sales order for customer X with products A and B", "create_sales_order"),
    ("Create a purchase order with supplier Z for product E", "create_purchase_order"),
    ("Place a purchase order with supplier W for products F and G", "create_purchase_order"),
    ("How many units of FRISOLAC are in stock?", "check_stock"),
    ("Order items from supplier M with quantity 5 of product J", "create_purchase_order"),
    ("Sales order for customer A with products T and U", "create_sales_order"),
    ("Purchase order with supplier B for products V, W", "create_purchase_order"),
    ("Order product M for customer O", "create_sales_order"),
    ("Place a purchase order for 10 units of product P with supplier Q", "create_purchase_order"),
    ("I need to create a sales order for customer J with products H, I", "create_sales_order"),
    ("I want to check stock availability for product ABC", "check_stock"),
    ("How much inventory do we have for product XYZ?", "check_stock"),
    ("Generate a purchase order for 20 chairs for supplier DEF", "create_purchase_order"),
    ("Create a sales order for 5 items of product GHI", "create_sales_order"),
    ("How many units are left of product A?", "check_stock"),
    ("Check inventory for product ABC", "check_stock"),
    ("What is the available quantity of FRISO?", "check_stock"),
    ("Check how much of product X is in warehouse", "check_stock"),
    ("Create a sales order for customer ABC with 5 boxes of FRISO", "create_sales_order"),
    ("I want to place a sales order", "create_sales_order"),
    ("Can you help me create an SO?", "create_sales_order"),
    ("Make a sales order for customer Peter", "create_sales_order"),
    ("Order 10 units of FRISO for client X", "create_sales_order"),
    ("Add a new sale for Glico product", "create_sales_order"),
    ("Sales order: customer Y needs 2 cases of milk", "create_sales_order"),
    ("Register new sales order for 'PLUS ONE'", "create_sales_order"),
    ("Place an order to sell 15 items of ABC", "create_sales_order"),
    ("Create a purchase order from supplier M&B", "create_purchase_order"),
    ("Order 100 bottles of hand sanitizer from supplier XYZ", "create_purchase_order"),
    ("Place a PO for 20 units of product DEF", "create_purchase_order"),
    ("I want to buy stock for Glico milk", "create_purchase_order"),
    ("Generate PO for 30 units from ABC supplier", "create_purchase_order"),
    ("New purchase order: 10 cartons of FRISO", "create_purchase_order"),
    ("I need to restock product P by ordering from vendor V", "create_purchase_order"),
    ("Send PO to supplier for 100 face masks", "create_purchase_order"),
    ("Check the status of sales order S00032", "find_by_number"),
    ("Check the status of sales order S00032", "find_by_number"),
    ("What is the status of PO00018?", "find_by_number"),
    ("Find sales order S00045 for me", "find_by_number"),
    ("Show details for order number PO123456", "find_by_number"),
    ("Get information about sales order SO_2023_1234", "find_by_number"),

    # find_by_partner samples
    ("List all orders from supplier ABC Corp", "find_by_partner"),
    ("Show purchase orders for vendor XYZ", "find_by_partner"),
    ("Find orders for customer John Doe", "find_by_partner"),
    ("Display sales orders for Glico company", "find_by_partner"),
    ("Where is the settings?", "navigate_help"),
    ("How do I open product page?", "navigate_help"),
    ("Can I automate tasks in Odoo?", "general_question"),
    ("What is the Sales module?", "navigate_help"),
    ("Is there an AI assistant for Odoo?", "general_question"),
    ("Can Odoo be used on mobile?", "general_question"),
    ("Is Odoo open source?", "general_question"),
    ("Who built Odoo?", "general_question"),
    ("Can I integrate Odoo with Gmail?", "general_question"),
    ("How much does Odoo cost?", "general_question"),
    ("Is Odoo suitable for small businesses?", "general_question"),
    ("How do I get to the Purchase Order module?", "navigate_help"),
    ("Where can I find the inventory dashboard?", "navigate_help"),
    ("Guide me to the Reporting section", "navigate_help"),
    ("Open the Sales menu", "navigate_help"),
    ("I need help finding the product list", "navigate_help"),
    ("Show me how to navigate to customers", "navigate_help"),
    ("What's the weather like today?", "irrelevant_question"),
    ("Tell me a joke", "irrelevant_question"),
    ("How tall is Mount Everest?", "irrelevant_question"),
    ("Translate 'hello' to French", "irrelevant_question"),
    ("Play some music", "irrelevant_question"),
    ("Look up sales order S00231", "find_by_number"),
    ("I want the details of PO000123", "find_by_number"),
    ("Search for order number SO000789", "find_by_number"),
    ("Retrieve info on sales order SO_321", "find_by_number"),
    ("Find me purchase order number PO999", "find_by_number"),
    ("Track status of SO1123", "find_by_number"),
    ("Show details for order S09999", "find_by_number"),
    ("Check PO00045 for status", "find_by_number"),
    ("Order SO1001, what’s its status?", "find_by_number"),
    ("What's happening with sales order S000999?", "find_by_number"),

    # irrelevant_question
    ("What's 2 plus 2?", "irrelevant_question"),
    ("Who is the president of France?", "irrelevant_question"),
    ("Tell me a fun fact", "irrelevant_question"),
    ("Recommend me a movie", "irrelevant_question"),
    ("Sing a song", "irrelevant_question"),
    ("What time is it?", "irrelevant_question"),
    ("Define artificial intelligence", "irrelevant_question"),
    ("How do airplanes fly?", "irrelevant_question"),
    ("Where is Paris?", "irrelevant_question"),
    ("What’s 5 squared?", "irrelevant_question"),

    # navigate_help
    ("Guide me to the settings page", "navigate_help"),
    ("Open Inventory module", "navigate_help"),
    ("Navigate to Sales", "navigate_help"),
    ("How do I go to the Reporting section?", "navigate_help"),
    ("I want to check vendor list", "navigate_help"),
    ("Help me reach the dashboard", "navigate_help"),
    ("Go to the Purchase module", "navigate_help"),
    ("How do I access product variants?", "navigate_help"),
    ("Show me the main menu", "navigate_help"),
    ("Take me to invoices", "navigate_help"),
    # find_by_partner
    ("List all orders from supplier ABC Corp", "find_by_partner"),
    ("Show purchase orders for vendor XYZ", "find_by_partner"),
    ("Find orders for customer John Doe", "find_by_partner"),
    ("Display sales orders for Glico company", "find_by_partner"),
    ("Show me all POs from supplier M&B", "find_by_partner"),
    ("List sales orders made for customer Peter", "find_by_partner"),
    ("Orders placed by vendor Johnson Ltd?", "find_by_partner"),
    ("What orders were created for client ABC?", "find_by_partner"),
    ("Find every order associated with customer Y", "find_by_partner"),
    ("Show all past orders for supplier Glico", "find_by_partner"),
    ("Search purchase history from vendor Hello Co.", "find_by_partner"),
    ("Give me sales orders linked to customer Alice", "find_by_partner"),
    ("Check orders involving vendor SupplyChain Inc", "find_by_partner"),
    ("Get list of orders from buyer Cường Hưng", "find_by_partner"),
    # checkstock
    ("How many units of product X are currently in stock?", "check_stock"),
    ("What’s the inventory level for item Y?", "check_stock"),
    ("Show me the stock quantity for SKU 12345.", "check_stock"),
    ("Can you check the available stock of Blue T-Shirt?", "check_stock"),
    ("I need to know how many pieces we have of product Z.", "check_stock"),
    ("What is the current on‑hand quantity for item ABC?", "check_stock"),
    ("Give me the remaining stock for orderable item #789.", "check_stock"),
    ("Tell me the warehouse stock level for Red Mug.", "check_stock"),
    ("Check inventory for product code M-001.", "check_stock"),
    ("How many do we have left of the Large Widget?", "check_stock"),
    ("Display available quantity for part number PN‑456.", "check_stock"),
    ("Retrieve current stock count for Green Jacket.", "check_stock"),
    ("What’s the stock on hand for item Bolt 8mm?", "check_stock"),
    ("Show available inventory for Printer Cartridge.", "check_stock"),
    ("Please report the current stock level of USB Cable.", "check_stock"),
    # fieldservice
    ("Create a field service task for customer XYZ on 2025-05-11", "create_field_service"),
    ("I need to schedule Field Service for Partner A with activity Installation", "create_field_service"),
    ("Please open a field service order for Company B, activity Maintenance", "create_field_service"),
    ("Generate a field service task assigned to me for customer C", "create_field_service"),
    ("Create a field service task for customer ACME Corp with activity Installation on 2025-06-01",
     "create_field_service"),
    ("Please schedule a Field Service order for customer Beta LLC, activity Maintenance, assigned to me",
     "create_field_service"),
    ("I need to open a field service job for customer Gamma Inc, activity Inspection, planned_date 2025-06-05",
     "create_field_service"),
    ("Generate a Field Service record for customer Delta Co, activity Repair, assign to user ‘jdoe’",
     "create_field_service"),
    ("Set up a field service task for customer Epsilon Ltd with activity Calibration", "create_field_service"),
    ("Please create a field service order for customer Zeta Enterprises on 2025-06-10 assigned to admin",
     "create_field_service"),

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
            ('svc', SVC(kernel='linear', probability=True)),
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
