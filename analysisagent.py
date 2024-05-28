import os
import openai
import getpass
import pandas as pd
import streamlit as st
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("Data Agent")

api_key = st.text_input("Enter your API Key:", type="password")

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
    
    response2 = data_analysis_agent.run("Group customers based on their country and plot a graph for the top 5 countries with most customers.")
    st.write(response2)
else:
    st.write("Please enter your OpenAI API Key to proceed.")
