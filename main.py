import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Load the dataset (Ensure the file is in the same directory as this script)
file_path = "car_price_dataset.csv"  
df = pd.read_csv(file_path)

# Ensure 'Price' column exists before categorization
if 'Price' in df.columns:
    df['Price_Category'] = pd.qcut(df['Price'], q=3, labels=['Low', 'Medium', 'High'])
else:
    raise ValueError("Column 'Price' not found in the dataset.")

# Selecting features (Check all exist in dataset)
features = ['Year', 'Engine_Size', 'Fuel_Type', 'Transmission', 'Mileage', 'Doors', 'Owner_Count']
missing_features = [f for f in features if f not in df.columns]

if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

target = 'Price_Category'

# Encoding categorical variables
categorical_cols = ['Fuel_Type', 'Transmission']
encoder = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])
    else:
        raise ValueError(f"Column '{col}' not found in dataset.")

# Scaling numerical variables
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Splitting data into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models with tuned hyperparameters
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
    "Random Forest_V2": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42),
    "Gradient Boosting_V2": GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
}

# Train and evaluate models
conf_matrices = {}
classification_reports = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
    conf_matrices[name] = cm
    
    # Generate classification report
    classification_reports[name] = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)

# Display confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjusting for 6 models
axes = axes.flatten()

for idx, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'], ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Generate summary report
summary_data = []

for name, report in classification_reports.items():
    accuracy = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]
    summary_data.append([name, accuracy, precision, recall, f1_score])

# Create a DataFrame for summary
df_summary = pd.DataFrame(summary_data, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])

# Print summary table
print("\nModel Performance Summary:")
print(df_summary)