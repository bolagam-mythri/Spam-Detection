📄 SMS Spam Detection with Logistic Regression
🚀 OverviewThis project focuses on building a machine learning model to classify SMS messages as Spam or Not Spam. Using text preprocessing, TF-IDF vectorization, and a Logistic Regression model, the project achieves accurate results on a public dataset.

🗂️ Project StructureFiles:spam_detection.ipynb: The Jupyter Notebook containing the full implementation of the project.
spam.csv: Dataset containing SMS messages labeled as Spam or Ham (Not Spam).
README.md: Project documentation (this file).

📊 DatasetSource: UCI Machine Learning Repository
Rows: 5574 messages
Columns:
v1: Label (ham or spam)
v2: SMS Message

⚙️ WorkflowData Cleaning:
Removed special characters and converted text to lowercase.
Mapped labels to binary format (0 for Ham, 1 for Spam).
Text Vectorization:
Applied TF-IDF vectorization with n-grams for feature extraction.
Model Training:
Used Logistic Regression for classification.
Split data into 80% training and 20% testing.
Evaluation:
Accuracy: ~98%
Classification report with Precision, Recall, and F1-Score.


🛠️ RequirementsInstall the following libraries to run the project:
pip install pandas scikit-learn numpy

📌 ResultsExample Predictions:MessagePredictionConfidence (Spam Probability)Hi, are we meeting today?Not Spam0.12Click to win a free cruise!Spam0.87Reminder: your class starts at 10 AMNot Spam0.15Congratulations, you've won a lottery!Spam0.95

🧪 How to RunClone this repository:
git clone https://github.com/bolagam-mythri/spam-detection.gitNavigate to the directory:
cd spam-detectionOpen the notebook in Jupyter or Colab:
jupyter notebook spam_detection.ipynbRun all cells to train the model and test predictions.


📖 Future WorkExperiment with advanced models like BERT or LSTM.
Use larger and more diverse datasets.
Deploy the model as a web app or API using Flask or FastAPI.


💡 AcknowledgmentsDataset: UCI SMS Spam Collection
Libraries: Scikit-learn, Pandas, Numpy


🌟 ConnectFor questions or feedback, feel free to reach out at:
Email: bolagammythri@gmail.com
GitHub: bolagam-mythri # Spam-Detection
