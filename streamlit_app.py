import streamlit as st
import joblib #buat import pickle
import pandas as pd

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model, user_input):
  prediction = model.predict([user_input])
  return prediction[0]

def main():
  st.title('Machine Learning App')  
  st.info('This app will predict your obesity level!')
  df = pd.read_csv("https://raw.githubusercontent.com/jackielim7/md-assign/master/ObesityDataSet_raw_and_data_sinthetic.csv")
  
  with st.expander("**Data :**"):
      st.write("This is a raw data")
      st.dataframe(df)

if __name__ == "__main__":
  main()

