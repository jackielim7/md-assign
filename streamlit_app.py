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

  with st.expander("**Data Visualization**"):
      # Matplotlib figure
      fig, ax = plt.subplots(figsize=(8, 6))
  
      # Create scatter plot
      scatter = sns.scatterplot(
          x=df[Height], 
          y=df[Weight], 
          hue=df[NObeyesdad], 
          palette="Set2", 
          s=200,  # Circle size
          edgecolor=None,  # No border on circles
          ax=ax
      )
  
      # Set axes to start from (0,0)
      ax.set_xlim(0, df[Height].max() + 1)
      ax.set_ylim(0, df[Weight].max() + 1)
  
      # Move legend below the plot with no border
      legend = ax.legend(title=hue_column, bbox_to_anchor=(0.5, -0.1), loc="upper center", frameon=False, ncol=2)
  
      # Display plot in Streamlit
      st.pyplot(fig)
  
if __name__ == "__main__":
  main()

