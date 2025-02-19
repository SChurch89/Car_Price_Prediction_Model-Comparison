# 🚀 Comparing Machine Learning Models for Car Price Classification

This project evaluates multiple **machine learning models** to classify car prices into categories (**Low, Medium, High**) using a dataset of vehicle attributes. The goal is to identify the best-performing model through hyperparameter tuning and model comparison.

---

## 📌 **Project Overview**
- The dataset contains car specifications such as `Year`, `Engine Size`, `Fuel Type`, `Transmission`, `Mileage`, `Doors`, and `Owner Count`.
- The target variable `Price` is categorized into three classes (`Low`, `Medium`, `High`) using **quantile-based binning**.
- Various **ML models** are trained, including:
  - ✅ **Decision Tree**
  - ✅ **Neural Network**
  - ✅ **Random Forest**
  - ✅ **Gradient Boosting**
  - ✅ **Random Forest_V2** (Tuned)
  - ✅ **Gradient Boosting_V2** (Tuned)

- Performance metrics such as **Accuracy, Precision, Recall, and F1-score** are compared.
- **Confusion matrices** are plotted for visualization.

---

## 🔧 **Setup & Installation**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## 2️⃣ Install Dependencies
Ensure you have Python installed. Then, install the required libraries:

```sh
pip install pandas scikit-learn matplotlib seaborn
```
## 3️⃣ Add the Dataset
Place your dataset car_price_dataset.csv inside the project directory.
🏃 Run the Script
Execute the script from the terminal:

```sh
python model_evaluation.py
```
## 🏃 Run the Script
Execute the script from the terminal:

```
python model_evaluation.py
```


## 📊 Results & Insights
The Gradient Boosting_V2 model achieved the highest accuracy (95.75%), followed by Neural Network (98.7%) and Random Forest_V2 (91.7%).

- Hyperparameter Tuning Improvements
- Random Forest vs Random Forest_V2:
- Improved accuracy from 90.75% ➝ 91.7% by increasing estimators (50 ➝ 100) and setting max_depth=15, min_samples_split=5.
- Gradient Boosting vs Gradient Boosting_V2:
- Accuracy increased from 87.0% ➝ 95.75% by raising the learning rate (0.1 ➝ 0.5) and boosting the number of estimators (50 ➝ 100).
🔥 Best Performing Model
Gradient Boosting_V2 achieved the highest accuracy and F1-score, making it the best choice for car price classification.

Random Forest vs Random Forest_V2:
Improved accuracy from 90.75% ➝ 91.7% by increasing estimators (50 ➝ 100) and setting max_depth=15, min_samples_split=5, which enhanced the model’s ability to generalize without overfitting.
Gradient Boosting vs Gradient Boosting_V2:
Accuracy increased from 87.0% ➝ 95.75% by raising the learning rate (0.1 ➝ 0.5) and boosting the number of estimators (50 ➝ 100), which allowed the model to converge faster and make stronger corrections at each boosting step.
Among all models, Neural Network (98.7%) had the second-highest accuracy, suggesting that deep learning methods performed well, though they may require longer training times.

🔥 Gradient Boosting_V2 stands out as the best model in this comparison due to its strong precision, recall, and F1-score, making it the most reliable choice for classification in this scenario. 🚀
