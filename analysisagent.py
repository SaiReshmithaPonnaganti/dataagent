import os
import openai
import getpass
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("Data Agent")

api_key = st.text_input("Enter your API Key:", type="password")

#function to check for graph
def is_graph(answer):
    keywords = ['plot', 'graph', 'visualize']
    for keyword in keywords:
        if keyword in answer.lower():
            return True
    return False

# Ensure the script waits for the user to input the API key
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

    # Load the dataset
    df = pd.read_csv("customers.csv")
    st.write(df.head())
    st.write(df.info())

    # Creating the OpenAI instance with the API key
    openai_instance = OpenAI(temperature=0, openai_api_key=api_key)

    # Creating the agent
    data_analysis_agent = create_pandas_dataframe_agent(openai_instance, df, verbose=True)

    # Running the agent
    st.write("Analyzing the data...")
    
    response1 = data_analysis_agent.run("List the columns with missing values and provide the best solution on how to handle them.")
    st.write(response1)
    
    user_input = st.text_input("Enter some text:")
    answer = data_analysis_agent.run(user_input)
    result=is_graph(answer)
    if (result==True):
        country_counts = df['Country'].value_counts()
        # Select the top 5 countries with the most customers
        top_5_countries = country_counts.head(5)
        # Plotting the graph
        st.write("Plotting the graph for the top 5 countries with the most customers...")
        fig, ax = plt.subplots()
        top_5_countries.plot(kind='bar', ax=ax)
        ax.set_title('Top 5 Countries with Most Customers')
        ax.set_xlabel('Country')
        ax.set_ylabel('Number of Customers')
        st.pyplot(fig)
    else:
        st.write(answer)

    
    response2 = data_analysis_agent.run("Group customers based on their country and plot a graph for the top 5 countries with most customers.")
    st.write(response2)
    


else:
    st.write("Please enter your OpenAI API Key to proceed.")