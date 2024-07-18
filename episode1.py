# pip install langchain_openai langchain_core

# Import the necessary modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the language model
llm = ChatOpenAI(api_key="...")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

# Create a function to accept user input and invoke the language model
def get_response(user_input):
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input})

# Get user input and print the response
user_input = input("Enter your text: ")
response = get_response(user_input)
print("Result:", response)