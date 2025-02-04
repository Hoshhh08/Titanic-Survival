Titanic Survival Prediction Model - README

Project Overview:
This project is a Titanic survival prediction model deployed using Streamlit. It allows users to input various passenger attributes and predict whether they would have survived the Titanic disaster based on a trained logistic regression model.

Files in the Deployment
app.py - Main Streamlit application for user interaction and prediction.
trained_logi_model.pkl - Pre-trained logistic regression model used for making predictions.
scaler.pkl - StandardScaler object used for feature scaling before prediction.

How to Use the Application:
Open the application in your browser after running the command above.

Click on the "Select Features" pop-up to adjust passenger attributes:
1.Age
2.Number of Siblings/Spouses Aboard
3.Number of Parents/Children Aboard
4.Gender
5.Embarkation from Cherbourg
6.WealthScore (calculated as Pclass × Fare)

-Click "Apply Features" to save the selected values.
-Click "Predict Survival" to get the survival prediction.

Model Details:
Algorithm: Logistic Regression
Features Used: Age, SibSp, Parch, Sex, Embarked_C, WealthScore
Preprocessing: Standardization using StandardScaler

Notes:
Ensure all necessary .pkl files (trained_logi_model.pkl and scaler.pkl) are present in the working directory.
The model predicts survival as "Survived 🎉" or "Sadly, Did Not Survive 😔".
The WealthScore feature represents a combination of Pclass and Fare, acting as a measure of socioeconomic status.

Author
Hoshang Sheth