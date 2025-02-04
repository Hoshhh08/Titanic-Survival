import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler

# Load trained model file:
with open('trained_logi_model.pkl', 'rb') as model_file:
    model = pkl.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pkl.load(scaler_file)

# Streamlit UI Setup:
st.set_page_config(page_title='Survival Quest: Titanic Edition', layout='centered')

# Add Background GIF (Replace 'background.gif' with your file path or URL)
gif_url = "https://i.pinimg.com/originals/8a/17/ba/8a17baa38518709469915741d11cea1a.gif"  # Update with your actual GIF file path or URL
# Inject CSS for Background GIF
st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] > .main {{
            background: url("{gif_url}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Titanic Survival Predictor')
st.markdown('### Choose the features from the pop-up window to predict survival.')

# Initialize session state for feature values:
if 'features' not in st.session_state:
    st.session_state.features = {
        'Age': 20,
        'SibSp': 0,
        'Parch': 0,
        'Sex_female': 0,
        'Embarked_C': 0,
        'WealthScore': 75,  # Default Value of Pclass*fare
    }

# Pop-up for Feature Selection:
with st.popover('Select Features'):
    age = st.slider('Age', 1, 100, st.session_state.features['Age'])
    sibsp = st.number_input('Siblings/Spouses Aboard', 0, 10, st.session_state.features['SibSp'])
    parch = st.number_input('Parents/Children Aboard', 0, 10, st.session_state.features['Parch'])
    gender = st.radio('Gender', ['Male', 'Female'])
    embarked = st.radio('Embarked at Cherbourg?', ['No', 'Yes'])
    wealthscore = st.number_input('WealthScore[Pclass*Fare]', 0, 600, st.session_state.features['WealthScore'])

    if st.button('Apply Features'):
        st.session_state.features = {
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Sex_female": 1 if gender == "Female" else 0,
            "Embarked_C": 1 if embarked == "Yes" else 0,
            "WealthScore": wealthscore,
        }
        st.rerun()  # Refresh UI to apply changes

# Display Selected Features
st.write('### Selected Features:')
for key, value in st.session_state.features.items():
    st.write(f'- **{key}:** {value}')

# Prediction Button
if st.button('Predict Survival'):
    # Convert features into DataFrame
    input_data = pd.DataFrame([st.session_state.features])

    # Standardize features
    input_scaled = scaler.transform(input_data)

    # Predict Survival
    prediction = model.predict(input_scaled)[0]

    # Display result
    result = "Survived 🎉" if prediction == 1 else "Sadly, Did Not Survive 😔"
    st.subheader(f"Prediction: {result}")
