import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

  with st.expander("Click to expand/minimize"):
      st.write("### Scatter Plot of Height vs. Weight by Obesity Level")
          # Create scatter plot
      fig, ax = plt.subplots(figsize=(8, 5))
      scatter = sns.scatterplot(
        data=df, 
        x="Height", 
        y="Weight", 
        hue="NObeyesdad", 
        palette="Set2", 
        s=100, 
        edgecolor="black"
      )
      # Improve legend placement
      legend = plt.legend(
        title="NObeyesdad",
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        borderaxespad=0
      )
          
      # Set labels & title
      plt.xlabel("Height")
      plt.ylabel("Weight")
      plt.title("Data Visualization", fontsize=12, fontweight="bold")

    # Show plot in Streamlit
  st.pyplot(fig)
  
if __name__ == "__main__":
  main()

