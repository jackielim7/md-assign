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
  pd.read_csv("https://github.com/jackielim7/md-assign/blob/master/ObesityDataSet_raw_and_data_sinthetic.csv")
  
  st.subheader("📊 Data")
  st.write("**Description:** This is a raw data")
  st.dataframe(df)

if __name__ == "__main__":
  main()

