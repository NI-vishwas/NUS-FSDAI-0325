from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_generate_pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def generate_text(input: TextInput):
    result = text_generate_pipeline(input.text, max_new_tokens=512)
    return {"label": result[0]["generated_text"][-1]['content'], "score": result[0]['score']}