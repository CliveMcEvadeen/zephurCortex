from llama_index.core import get_response_synthesizer
from llama_index.response_synthesizers import TreeSummarize

# Initialize the response synthesizer
response_synthesizer = TreeSummarize(verbose=True)

# Define the prompt template
prompt_template = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Create a prompt object
prompt = prompt_template(prompt_template)

# Set the context and query
context = "This is some context information."
query = "What is the meaning of life?"

# Get the response
response = response_synthesizer.get_response(query, [context], prompt=prompt)

print(response)