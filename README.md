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

2️⃣ Install Dependencies
Ensure you have Python installed. Then, install the required libraries:

sh
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn
3️⃣ Add the Dataset
Place your dataset car_price_dataset.csv inside the project directory.
🏃 Run the Script
Execute the script from the terminal:

sh
Copy
Edit
python model_evaluation.py
📊 Results & Insights
The Gradient Boosting_V2 model achieved the highest accuracy (95.75%), followed by Neural Network (98.7%) and Random Forest_V2 (91.7%).

Hyperparameter Tuning Improvements
Random Forest vs Random Forest_V2:
Improved accuracy from 90.75% ➝ 91.7% by increasing estimators (50 ➝ 100) and setting max_depth=15, min_samples_split=5.
Gradient Boosting vs Gradient Boosting_V2:
Accuracy increased from 87.0% ➝ 95.75% by raising the learning rate (0.1 ➝ 0.5) and boosting the number of estimators (50 ➝ 100).
🔥 Best Performing Model
Gradient Boosting_V2 achieved the highest accuracy and F1-score, making it the best choice for car price classification.

📜 Project Structure
bash
Copy
Edit
📂 your-repository/
 ├── 📄 model_evaluation.py   # Main script for training and evaluation
 ├── 📄 car_price_dataset.csv  # Dataset (add manually)
 ├── 📄 README.md              # Project documentation
🤝 Contributing
Feel free to fork this repo, improve the models, and submit a pull request! 🚀

📜 License
This project is open-source and free to use under the MIT License.

yaml
Copy
Edit

---

### ✅ **How to Use**
1. Save this as **`README.md`** in your project root.
2. Commit and push to GitHub:
   ```sh
   git add README.md
   git commit -m "Added README file"
   git push origin main
This README makes your project professional, easy to understand, and contributor-friendly! 🎯 Let me know if you want any modifications. 🚀