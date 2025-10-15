import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import joblib

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model, user_input):
  probabilities = model.predict_proba([user_input])
  return probabilities

def main():
  st.title('Machine Learning App')  
  st.info('This app will predict your obesity level!')

  #Read data
  df = pd.read_csv("https://raw.githubusercontent.com/jackielim7/md-assign/master/ObesityDataSet_raw_and_data_sinthetic.csv")
  
  #Expand raw data
  with st.expander("**Data**"):
      st.write("This is a raw data")
      st.dataframe(df)

  #Expand visualization
  with st.expander("**Data Visualization**"):
      #Interactive scatter plot
      fig = px.scatter(df, 
                       x="Height", 
                       y="Weight", 
                       color="NObeyesdad",
                       labels={"Height": "Height", "Weight": "Weight"},
                       template="plotly_white")

      fig.update_xaxes(range=[0, 2.1], dtick=0.1, tickangle=0)
      fig.update_yaxes(range=[0, 180], dtick=20)

      #Legend
      fig.update_layout(legend=dict(
          orientation="h",
          yanchor="top", 
          y=-0.2, 
          xanchor="center", 
          x=0.5
      ))

      #Show plot
      st.plotly_chart(fig)

  Gender = st.selectbox('Gender', ('Male', 'Female')) 
  Age = st.slider('Age', min_value = 1, max_value = 80, value = 1)
  Height = st.number_input('Height (meters)', min_value=1.0, max_value=3.0, value=1.0, step=0.01) 
  Weight = st.number_input('Weight (kilograms)', min_value=1.0, max_value=200.0, value=1.0, step=0.1)
  family_history_with_overweight = st.selectbox('family_history_with_overweight', ('yes', 'no'))
  FAVC = st.selectbox('FAVC', ('yes', 'no'))
  FCVC = st.number_input('FCVC', min_value=1.0, max_value=3.0, value=1.0, step=0.1) 
  NCP = st.number_input('NCP', min_value=1.0, max_value=4.0, value=1.0, step=0.1) 
  CAEC = st.selectbox('CAEC', ('Always', 'Frequently', 'Sometimes', 'no'))
  SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
  CH2O = st.number_input('CH2O', min_value=1.0, max_value=3.0, value=1.0, step=0.1) 
  SCC = st.selectbox('SCC', ('yes', 'no'))
  FAF = st.number_input('FAF', min_value=0.0, max_value=3.0, value=0.0, step=0.1)
  TUE = st.number_input('TUE', min_value=0.000, max_value=2.000, value=0.000, step=0.01)
  CALC = st.selectbox('CALC', ('Always', 'Frequently', 'Sometimes', 'no'))
  MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

 #Encode categorical
  gender_map = {'Male': 0, 'Female': 1}
  binary_map = {'no': 0, 'yes': 1}
  caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
  calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

  #OHE
  mtrans_categories = ['MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
  mtrans_encoded_dict = {col: 1 if MTRANS in col else 0 for col in mtrans_categories}

  #Save raw data for display
  raw_data = {
    "Gender": Gender,
    "Age": Age,
    "Height": Height,
    "Weight": Weight,
    "family_history_with_overweight": family_history_with_overweight,
    "FAVC": FAVC,
    "FCVC": FCVC,
    "NCP": NCP,
    "CAEC": CAEC,
    "SMOKE": SMOKE,
    "CH2O": CH2O,
    "SCC": SCC,
    "FAF": FAF,
    "TUE": TUE,
    "CALC": CALC,
    "MTRANS": MTRANS
  }

  #Display raw data (user input)
  raw_df = pd.DataFrame([raw_data])
  st.write("**Data Input by User**")
  st.dataframe(raw_df)

  #Save encode version
  user_input = [
      gender_map[Gender],
      Age,
      Height,
      Weight,
      binary_map[family_history_with_overweight],
      binary_map[FAVC],
      FCVC,
      NCP,
      caec_map[CAEC],
      binary_map[SMOKE],
      CH2O,
      binary_map[SCC],
      FAF,
      TUE,
      calc_map[CALC],
  ] + list(mtrans_encoded_dict.values())

  #Load model and predict
  model_filename = 'trained_model.pkl'
  model = load_model(model_filename)
  prediction_proba = predict_with_model(model, user_input)
  
  #Convert prediction into df
  class_labels = model.classes_
  prediction_df = pd.DataFrame(prediction_proba, columns=class_labels).round(4)  #Round values for better readability
  
  #Display prediction df
  st.write("**Obesity Prediction**")
  st.dataframe(prediction_df)

  #Final
  predicted_class = class_labels[prediction_proba.argmax()]
  
  #Display Final Predicted Class
  st.write(f"**The predicted output is: `{predicted_class}`**")
  
if __name__ == "__main__":
    main()
