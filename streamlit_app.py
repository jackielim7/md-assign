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

    
if __name__ == "__main__":
  main()

