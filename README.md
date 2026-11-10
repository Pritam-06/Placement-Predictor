# Placement-Predictor

📊 Student Placement Prediction using Machine Learning
A beginner-friendly Machine Learning project that predicts whether a student will be placed based on their CGPA and IQ scores.

This project demonstrates an end-to-end Machine Learning workflow — from data loading and exploratory data analysis to preprocessing, feature scaling, model training, evaluation, decision-boundary visualization, and model serialization.
📚 Table of Contents
- Overview 
- Project Structure
- Dataset
- Machine Learning Workflow   
- Tech Stack   
- Getting Started
- Model Performance
- Model Usage
- Future Improvements
- Author
📋 Overview
This project uses Logistic Regression to classify students into two categories:
- 1 → Placed
- 0 → Not Placed
The model uses two features:
- CGPA
- IQ
The dataset contains 100 student records.

The project was created to understand the complete workflow of building and evaluating a basic classification model using Python and scikit-learn.
📁 Project Structure
├── placement.csv          # Student placement dataset
├── Toy_Project.ipynb      # Jupyter Notebook containing the complete ML workflow
├── placement.pkl          # Trained Logistic Regression model
└── README.md              # Project documentation
Note: placement.pkl is generated when the model-saving cell in the notebook is executed.

📊 Dataset
The dataset (placement.csv) contains 100 student records with the following columns:
Column	Data Type	Description
Unnamed: 0	Integer	Index column removed during preprocessing
cgpa	Float	Student's Cumulative Grade Point Average
iq	Float	Student's Intelligence Quotient score
placement	Integer	Target variable (1 = Placed, 0 = Not Placed)


🎯 Features and Target
Independent Variables (X):
CGPA
IQ
Dependent Variable (y):
placement
🔄 Machine Learning Workflow
1. Data Loading and Inspection
The dataset is loaded using Pandas:
import pandas as pd

df = pd.read_csv("placement.csv")
The dataset is then inspected and the unnecessary index column is removed:
df = df.iloc[:, 1:]

2. Exploratory Data Analysis (EDA)
A scatter plot is used to visualize the relationship between CGPA, IQ, and the placement classes.

This helps understand how the two features are distributed and how well the classes can potentially be separated.

3. Feature and Target Extraction
The input features are separated from the target variable:
x = df.iloc[:, 0:2]
y = df.iloc[:, -1]
Here:
- x contains CGPA and IQ
- y contains the placement labels
4. Train-Test Split
The dataset is divided into training and testing sets using an 80:20 split:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
Using random_state=42 makes the split reproducible.
Important: The original notebook does not currently specify random_state=42. The code above is the recommended reproducible version. If you want the README to exactly document the current notebook, remove random_state=42 and note that the split may vary between runs.

5. Feature Scaling
Since CGPA and IQ have different numerical scales, StandardScaler is used to standardize the features.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
The scaler is fitted only on the training data and then applied to the test data.

6. Model Training
A Logistic Regression classifier is trained using the scaled training data:
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train)
Logistic Regression is suitable here because the project is performing binary classification.

7. Prediction
The trained model is used to predict the placement labels for the test set:
y_pred = clf.predict(x_test)

8. Model Evaluation
The model's performance is evaluated using accuracy:
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
The current project reports an accuracy of approximately 90% on the test split.
Note: Because the original notebook's train_test_split() does not use a fixed random_state, the exact accuracy can vary between runs. For reproducible results, use random_state=42.

9. Decision Boundary Visualization
The project uses mlxtend to visualize the decision regions learned by the Logistic Regression model:
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(
    x_train,
    y_train.values,
    clf=clf,
    legend=2
)
This provides a visual representation of how the trained classifier separates the two placement classes.
10. Model Serialization
The trained model is saved using Python's pickle module:
import pickle

pickle.dump(clf, open("placement.pkl", "wb"))
This allows the trained model to be saved and loaded later without retraining it.
🛠️ Tech Stack
Technology	Purpose
Python 3.x	Programming language
Pandas	Data loading and manipulation
NumPy	Numerical operations
Matplotlib	Data visualization
Scikit-learn	Preprocessing, model training, and evaluation
Mlxtend	Decision-boundary visualization
Pickle	Model serialization
Jupyter Notebook	Development environment


🚀 Getting Started
1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Replace your-username/your-repository-name with your actual GitHub repository URL.
2. Install Dependencies
Make sure Python 3.x is installed, then run:
pip install numpy pandas matplotlib scikit-learn mlxtend jupyter
3. Run the Notebook
Start Jupyter Notebook:
jupyter notebook
Then open:
Toy_Project.ipynb
Make sure placement.csv is in the same directory as the notebook.

You can also open the notebook using Google Colab.
📈 Model Performance
Metric	Value
Algorithm	Logistic Regression
Features	CGPA, IQ
Training Data	80%
Testing Data	20%
Test Accuracy	90%


