from fastapi import FastAPI, Request, Query # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore   
from typing import Optional, List
import os
import requests
from .search import search
import json

app = FastAPI()
# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: Optional[List[Link]] = None

# LLM Answering Logic
import os
import requests

def ask_llm_with_chunks(question, top_chunks, image_base64=None, model="mistralai/mistral-7b-instruct"):
    # api_key = os.environ.get("OPENAI_API_KEY")
    api_key = "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"
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Prepare message text
    chunk_texts = "\n\n".join(
    f"{i+1}. {chunk['content'].strip()}\n(Source: {chunk['url']})"
    if chunk.get("url") else
    f"{i+1}. {chunk['content'].strip()}"
    for i, chunk in enumerate(top_chunks)
    )
    # return chunk_texts
    user_message = [
        { "type": "text", "text": f"Here is a question:\n\n{question}\n\nHere are some related chunks:\n\n{chunk_texts}\n\nBased on the chunks above, give the best possible answer." }
    ]

    if image_base64:
        user_message.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        })

    # Prepare request
    payload = {
        "model": model,
        "max_tokens": 3700,
        "messages": [
            {
                "role": "system",
                "content": '''You are a helpful virtual TA for the Tools in Data Science course. Respond clearly and concisely.
                You will be given a question, an optional image and some related chunks of text. Use the best chunks to answer the question.
                output must follow the format:
                {
                    "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
                    "links": [
                        {
                        "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                        "text": "Use the model that’s mentioned in the question."
                        },
                        {
                        "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                        "text": "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
                        }
                    ]
                }
                answer is the main response to the question, and links is a list of relevant discourse links that can help the user understand the answer better.
                If you cannot find any relevant links, return an empty list.
                If the question is not related to the Tools in Data Science course, respond with "I am sorry, I cannot answer this question." and return an empty list for links.
                text in links should be a short description of the link, and url should be the actual discourse link.
                '''
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    # url = "https://aipipe.org/openrouter/v1/chat/completions"
    url = "https://openrouter.ai/api/v1/chat/completions"
    response = requests.post(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tds-virtual-ta-one-rouge.vercel.app"
        },
        json=payload
    )

    if response.status_code != 200:
        raise Exception(f"OpenAI error {response.status_code}: {response.text}")
    response_data = response.json()
    usage = response_data.get("usage", {})
    print(f"[Token usage] prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')}, total: {usage.get('total_tokens')}")

    return response_data["choices"][0]["message"]["content"]
    # return response.json()["choices"][0]["message"]["content"]
    # return chunk_texts

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-powered Q&A API. Use POST /api/ to ask questions."}
@app.post("/api/")
async def handle_query(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        image = data.get("image")

        if not question:
            return {"error": "Missing 'question'"}

        top_chunks = [chunk for _, chunk in search(query=question, image_base64=image, top_k=5)]
        response = ask_llm_with_chunks(
            question=question,
            top_chunks=top_chunks,
            image_base64=image
        )

        return json.loads(response)

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# def handle_query(input: QueryRequest):
#     try:
#         top_chunks = [chunk for _, chunk in search(query=input.question, image_base64=input.image, top_k=5)]

#         response = ask_llm_with_chunks(
#             question=input.question,
#             top_chunks=top_chunks,
#             image_base64=input.image, # or any free OpenRouter model like "mistralai/mistral-7b-instruct"
#         )

#         return json.loads(response)

#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e)
#         }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)
## testing the ask_llm_with_chunks function ##
# question = "should i take this course?"
# top_chunks = [chunk for _, chunk in search(query=question, top_k=5)]

# print(ask_llm_with_chunks(
#     question=question,
#     top_chunks=top_chunks
# ))


















