# Salary-Anomaly-Detection-using-Isolation-Forest
💼 Anomaly Detection on Salary vs Working Hours
This project uses Isolation Forest to detect anomalies in a dataset containing salary (in INR) and working hours. It showcases two different approaches to anomaly detection using machine learning:

Method 1: Manual separation using direct predictions (fit_predict)

Method 2: Anomaly score-based detection using a custom threshold (decision_function)

📁 Dataset
The dataset used is:

salary_anomaly_dataset_modified.csv
Contains two columns:

Salary_INR: Monthly salary in Indian Rupees (e.g. Rs 562,178)

Working_Hours: Weekly working hours (e.g. 42.5 hrs)

Note: The dataset has missing values and string formatting which are handled during preprocessing.

📌 Key Features
Handles real-world messy data (e.g., currency strings and unit suffixes)

Implements and compares two Isolation Forest approaches

Custom contamination rate calculation

Anomaly visualization using scatter plots

Optionally evaluates model with Accuracy, Precision, Recall, and F1-score (if ground truth is available)

🛠️ Technologies Used
Python 3.x

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

🧪 How to Run
Clone this repository or download the source code.

Make sure you have all dependencies installed:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn seaborn
Place your dataset file salary_anomaly_dataset_modified.csv in the same directory.

Run the Python script:

bash
Copy
Edit
python anomaly_detection.py
🔍 Project Structure
bash
Copy
Edit
.
├── salary_anomaly_dataset_modified.csv
├── anomaly_detection.py
└── README.md
📊 Visual Output
Scatter plots of Salary vs Working Hours:

Blue: Normal data points

Red: Anomalies

Example Output:

<!-- Replace with actual image if hosted -->

📈 Optional Evaluation Metrics
If you have y_true labels:

python
Copy
Edit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
You can compute:

Accuracy

Precision

Recall

F1-score

✅ Summary of Methods
Feature	Method 1	Method 2
Detection	Uses fit_predict() (-1 & 1)	Uses decision_function() (score)
Threshold	None (default)	Custom threshold
Flexibility	Less	More
Metrics	Not built-in	Evaluation supported
🙌 Author
Made with ❤️ by [Your Name]
Feel free to connect or contribute!

Let me know if you want this tailored to Jupyter Notebook style or with more GitHub-style badges!



You said:
salary_anomaly_dataset_modified.csv , main.ipynb github link-https://github.com/majumdarjoyeeta/Salary-Anomaly-Detection-using-Isolation-Forest

ChatGPT said:
Perfect! Here's the updated README.md for your project, including your GitHub repository link and file references:

💼 Salary Anomaly Detection using Isolation Forest
This project applies Isolation Forest, an unsupervised machine learning algorithm, to detect anomalies in a dataset of salaries and working hours. It includes two distinct approaches for outlier detection, with visualizations and optional performance evaluation.

📍 GitHub Repository: https://github.com/majumdarjoyeeta/Salary-Anomaly-Detection-using-Isolation-Forest

📁 Files
salary_anomaly_dataset_modified.csv: Cleaned dataset with salary and working hours.

main.ipynb: Jupyter Notebook implementing both anomaly detection methods with detailed explanations and visualizations.

📌 Key Concepts
Isolation Forest for anomaly detection.

Data preprocessing: handling missing values, converting string formats (e.g. "Rs 562,178" to numeric).

Visualizing anomalies in salary vs. working hours.

Manual thresholding vs model-predicted contamination rates.

Optional model evaluation (Accuracy, Precision, Recall, F1-score).

🧠 Methods Used
1️⃣ Method 1: Manual Labeling with fit_predict()
Detects anomalies using direct model predictions (-1 = Anomaly, 1 = Normal).

Contamination rate is calculated from the results and used for retraining.

Simple and straightforward.

2️⃣ Method 2: Threshold-Based Scoring
Uses decision_function() to get anomaly scores.

Allows setting a custom threshold for identifying outliers.

More flexible and tunable.

📊 Sample Output
Scatter plot showing:

🔵 Normal data

🔴 Anomalies (outliers)

Optional metrics (if true labels are available):

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-score

🚀 How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/majumdarjoyeeta/Salary-Anomaly-Detection-using-Isolation-Forest.git
cd Salary-Anomaly-Detection-using-Isolation-Forest
Install the required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Open the notebook:

bash
Copy
Edit
jupyter notebook main.ipynb
🛠️ Tech Stack
Python 3.x

Jupyter Notebook

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

📎 License
This project is open-source and available under the MIT License.

🙌 Acknowledgments
Developed by @majumdarjoyeeta
Feel free to fork, star ⭐, or contribute!
