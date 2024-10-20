from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import streamlit as st
openai.api_key = "Open_Api_Key"

# Define your index name
index_name = "test-model"  # Replace with your desired index name

# Initialize Pinecone with the API key directly (for testing purposes)
pc = Pinecone(
    api_key="Pinecone_Api_Key"  # Replace with your actual Pinecone API key
)

# # Check if the index exists, if not, create it
# if index_name not in pc.list_indexes(): 
#     pc.create_index(
#         name=index_name, 
#         dimension=1536,  # This is the dimension for OpenAI's text-embedding-ada-002 model
#         metric='cosine',  # You can use 'euclidean' or 'dotproduct' as well
#         spec=ServerlessSpec(
#             cloud='aws',     # Choose the cloud provider (AWS in this case)
#             region='us-east-1'  # Choose the region (us-west-2 as an example)
#         )
#     )

# Connect to the index
index = pc.Index(index_name)

def find_match(input):
    # Step 1: Generate embedding for the input using OpenAI API
    input_embedding = openai.embeddings.create(
        input=[input],
        model="text-embedding-ada-002"  # Ensure you're using the correct embedding model
    ).data[0].embedding
    
    # Step 2: Query Pinecone using keyword arguments
    result = index.query(
        vector=input_embedding, 
        top_k=2, 
        namespace="",  # Provide a namespace if necessary
        include_metadata=True  # Ensure metadata is included in the result
    )
    
    # Step 3: Safely extract the text from the metadata of the top 2 matches
    match_1_text = result.matches[0].metadata.get('text', 'No metadata text available') if result.matches[0].metadata else 'No metadata available'
    match_2_text = result.matches[1].metadata.get('text', 'No metadata text available') if result.matches[1].metadata else 'No metadata available'
    
    # Step 4: Return the combined result of the two matches
    return match_1_text + "\n" + match_2_text



def query_refiner(conversation, query):

    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user queries based on conversation history."},
            {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
