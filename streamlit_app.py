import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

def load_model(filename):
  model = joblib.load(filename)
  return model

def predict_with_model(model, user_input):
  prediction = model.predict([user_input])
  return prediction[0]

def main():
  st.title('Machine Learning App')  
  st.info('This app will predict your obesity level!')

  #Read data
  df = pd.read_csv("https://raw.githubusercontent.com/jackielim7/md-assign/master/ObesityDataSet_raw_and_data_sinthetic.csv")
  
  #Expand buat bagian raw data
  with st.expander("**Data**"):
      st.write("This is a raw data")
      st.dataframe(df)

  # Expand for visualization
  with st.expander("**Data Visualization**"):
      # Create an interactive scatter plot
      fig = px.scatter(df, 
                       x="Height", 
                       y="Weight", 
                       color="NObeyesdad",
                       labels={"Height": "Height", "Weight": "Weight"},
                       template="plotly_white")

      fig.update_xaxes(range=[0, 2.1], dtick=0.1, tickangle=0)
      fig.update_yaxes(range=[0, 180], dtick=20)

      # Move legend below the plot
      fig.update_layout(legend=dict(
          orientation="h",  # Horizontal legend
          yanchor="top", 
          y=-0.2,  # Move it below the plot
          xanchor="center", 
          x=0.5
      ))

      # Show the interactive plot
      st.plotly_chart(fig)

  Gender = st.selectbox('Gender', ('Male', 'Female')) 
  Age = st.slider('Age', min_value = 1, max_value = 80, value = 1)  # ✅ Allow typing
  Height = st.number_input('Height (meters)', min_value=1.0, max_value=3.0, value=1.0, step=0.01)  # ✅ Allow decimals
  Weight = st.number_input('Weight (kilograms)', min_value=1.0, max_value=200.0, value=1.0, step=0.1)  # ✅ Allow decimals
  family_history_with_overweight = st.selectbox('family_history_with_overweight', ('yes', 'no'))
  FAVC = st.selectbox('FAVC', ('yes', 'no'))
  FCVC = st.number_input('FCVC', min_value=1.0, max_value=3.0, value=1.0, step=0.1) 
  NCP = st.number_input('NCP', min_value=1.0, max_value=4.0, value=1.0, step=0.1) 
  CAEC = st.selectbox('CAEC', ('Always', 'Frequently', 'Sometimes', 'no'))
  SMOKE = st.selectbox('SMOKE', ('yes', 'no'))
  CH2O = st.number_input('CH2O', min_value=1.0, max_value=3.0, value=1.0, step=0.1) 
  SCC = st.selectbox('SCC', ('yes', 'no'))
  FAF = st.number_input('FAF', min_value=0.0, max_value=3.0, value=0.0, step=0.1)
  TUE = st.number_input('TUE', min_value=0.000, max_value=2.000, value=0.000, step=0.001)
  CALC = st.selectbox('CALC', ('Always', 'Frequently', 'Sometimes', 'no'))
  MTRANS = st.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))


if __name__ == "__main__":
  main()

