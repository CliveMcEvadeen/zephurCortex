# # import transformers
# # import torch

# # model_id = "meta-llama/Meta-Llama-3-8B"

# # pipeline = transformers.pipeline(
# #     "text-generation", 
# #     model=model_id, 
# #     model_kwargs={"torch_dtype": torch.bfloat16}, 
# #     device_map="auto"
# # )

# # pipeline("Hey how are you doing today?")
# # pipeline("what is your name?")

# import os
# from openai import OpenAI

# client = OpenAI(
# api_key=os.getenv("API_TOKEN"),
# base_url="https://api.aimlapi.com",
#     )
# response = OpenAI.Completion.create(
# model="meta-llama/Llama-3-8b-chat-hf",
# messages=[
# {
# "role": "system",
# "content": "You are an AI assistant who knows everything.",
# },
# {
# }
# ],
# "role": "user",
# "content": "Tell me, why is the sky blue?"
# message = response['choices'][0]['text']
# print (f"Assistant: {message}")

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key='LL-DE2Qul2DJhG3voo8Pmof7Yr7hER1kmOa7lIdNMMH1FDWrSnwA6ixJ1tm4145E0zc',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="meta-llama/Llama-3-8b-chat-hf",
)
