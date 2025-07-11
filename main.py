import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX, GEMINI_API_KEY

# ============================ SETUP ============================
app = FastAPI()

# Google Gemini Setup
genai.configure(api_key=GEMINI_API_KEY)

# Pinecone Init
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX)
embedding_model = "models/embedding-001"

# ============================ MODELS ============================
class UserInput(BaseModel):
    query: str

# ============================ DATA ============================
df = pd.read_csv("indian_food_recipes.csv")
df = df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions'])

# ============================ EMBEDDING ============================
def embed(text):
    res = genai.embed_content(model=embedding_model, content=text, task_type="RETRIEVAL_DOCUMENT")
    return res['embedding']

# ============================ INDEXING ============================
@app.post("/index")
def index_recipes():
    to_upsert = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        content = f"Recipe: {row['TranslatedRecipeName']}\n\nIngredients: {row['TranslatedIngredients']}\n\nInstructions: {row['TranslatedInstructions']}\n\nCuisine: {row['Cuisine']}, Course: {row['Course']}, Diet: {row['Diet']}, Complexity: {row['ComplexityLevel']}"
        vec = embed(content)
        to_upsert.append((str(row['Srno']), vec, {"name": row['TranslatedRecipeName'], "main_ingredient": row['MainIngredient']}))
    index.upsert(vectors=to_upsert)
    return {"message": "Recipes indexed successfully!"}

# ============================ SEARCH ============================
def search_similar_recipes(query):
    query_vec = embed(query)
    results = index.query(vector=query_vec, top_k=5, include_metadata=True)
    return results['matches']

# ============================ GENERATION ============================
def generate_recipe(user_input, context):
    prompt = f"""
You are a culinary assistant AI. Generate a new Indian recipe using this user input:
"{user_input}"

Use the following relevant recipes as inspiration:
{context}

The recipe should include:
- Title
- Ingredients
- Instructions
- Cuisine
- Course
- Complexity level
- Main ingredient

Format your output cleanly.
"""
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
    response = model.generate_content(prompt)
    return response.text

# ============================ GENERATE ENDPOINT ============================
@app.post("/generate")
def generate(user_input: UserInput):
    try:
        matches = search_similar_recipes(user_input.query)
        context = "\n\n".join([f"{m['metadata']['name']} (Main Ingredient: {m['metadata']['main_ingredient']})" for m in matches])
        result = generate_recipe(user_input.query, context)
        return {"generated_recipe": result}
    except Exception as e:
        return {"error": str(e)}

