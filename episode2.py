import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "..."  # Replace with your OpenAI API key

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLM
llm = OpenAI(temperature=0.8)

# Chains
chain = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born"
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world"
)

chain3 = LLMChain(
    llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory
)

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob', 'description'], verbose=True
)

# Function to run the chain
def search_celebrity(name):
    result = parent_chain({'name': name})
    return result, person_memory.buffer, descr_memory.buffer

# Example usage
if __name__ == "__main__":
    celebrity_name = input("Enter the name of the celebrity: ")
    if celebrity_name:
        result, person_info, major_events = search_celebrity(celebrity_name)
        print("Result:", result)
        print("\nPerson Info:", person_info)
        print("\nMajor Events:", major_events)