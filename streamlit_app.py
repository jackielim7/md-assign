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

  # with st.expander("**Data Visualization**"):
  #     # Matplotlib figure
  #     fig, ax = plt.subplots(figsize=(10  , 5))
  
  #     # Create scatter plot
  #     scatter = sns.scatterplot(
  #         x=df["Height"], 
  #         y=df["Weight"], 
  #         hue=df["NObeyesdad"], 
  #         palette="Set1", 
  #         s=50,  # Circle size
  #         edgecolor=None,  # No border on circles
  #         ax=ax
  #     )
  
  #     # Set axes to start from (0,0)
  #     ax.set_xlim(0, 2)
  #     ax.set_xticks([i / 10 for i in range(0, 21)])
  #     ax.set_ylim(0, 180)
  
  #     # Move legend below the plot with no border
  #     legend = ax.legend(title="NObeyesdad", bbox_to_anchor=(0.5, -0.1), loc="upper center", frameon=False, ncol=2)
  
  #     # Display plot in Streamlit
  #     st.pyplot(fig)

  # Expand for visualization
  with st.expander("**Data Visualization**"):
      # Create an interactive scatter plot
      fig = px.scatter(df, 
                       x="Height", 
                       y="Weight", 
                       color="NObeyesdad",
                       title="Height vs Weight",
                       labels={"Height": "Height (m)", "Weight": "Weight (kg)"},
                       template="plotly_white")

      # Set X-axis ticks every 0.1
      fig.update_xaxes(
          tick0=0,  # Start at 0
          dtick=0.1,  # Interval of 0.1
          tickangle=0  # Keep labels horizontal
      )

      # Set Y-axis ticks every 20
      fig.update_yaxes(
          tick0=0,  # Start at 0
          dtick=20  # Interval of 20
      )

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

