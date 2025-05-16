
# RAG Recipe Generator using Pinecone + Google Gemini + Streamlit (Frontend)

# Requirements:
# pip install pandas pinecone-client google-generativeai streamlit

import pandas as pd
import streamlit as st
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX, GEMINI_API_KEY

# ============================ SETUP ============================
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

# ============================ LOAD DATA ============================
@st.cache_data
def load_data():
    df = pd.read_csv("indian_food_recipes.csv")
    df = df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions'])
    return df

df = load_data()

# ============================ EMBEDDING & UPSERT ============================
def embed(text):
    res = genai.embed_content(model=embedding_model, content=text, task_type="RETRIEVAL_DOCUMENT")
    return res['embedding']

@st.cache_resource
def index_recipes(df):
    to_upsert = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        content = f"Recipe: {row['TranslatedRecipeName']}\n\nIngredients: {row['TranslatedIngredients']}\n\nInstructions: {row['TranslatedInstructions']}\n\nCuisine: {row['Cuisine']}, Course: {row['Course']}, Diet: {row['Diet']}, Complexity: {row['ComplexityLevel']}"
        vec = embed(content)
        to_upsert.append((str(row['Srno']), vec, {"name": row['TranslatedRecipeName'], "main_ingredient": row['MainIngredient']}))
    index.upsert(vectors=to_upsert)

# Run once to embed and index recipes
if st.button("Index Recipes to Pinecone"):
    index_recipes(df)
    st.success("Recipes indexed successfully!")

# ============================ SEARCH + GENERATE ============================
def search_similar_recipes(query):
    query_vec = embed(query)
    results = index.query(vector=query_vec, top_k=5, include_metadata=True)
    return results['matches']

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

# ============================ FRONTEND ============================
st.title("🇮🇳 AI Indian Recipe Generator 🍛")

user_input = st.text_input("Enter ingredients or what you want to cook:")

if st.button("Generate Recipe") and user_input:
    st.info("Searching similar recipes...")
    matches = search_similar_recipes(user_input)
    context = "\n\n".join([f"{m['metadata']['name']} (Main Ingredient: {m['metadata']['main_ingredient']})" for m in matches])
    st.info("Generating new recipe using Gemini...")
    result = generate_recipe(user_input, context)
    st.markdown("---")
    st.markdown(result)

# ============================ END ============================

# Save this as app.py and run with: streamlit run app.py
# Make sure the CSV file "indian_food_recipes.csv" is in the same directory.
