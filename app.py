import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. Initialize API and Keys
load_dotenv()
app = FastAPI(title="Chef AI - 2026 Edition")

# 2. Load the Vector Database
# Ensure the model_name matches exactly what you used in ingest.py
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(
    "vector_db", 
    embeddings, 
    allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Setup the 2026 Brain (Gemini 3 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2
)

# 4. Request Schema
class RecipeRequest(BaseModel):
    ingredients: list[str]
    constraints: str = "None"

@app.post("/recommend")
async def recommend_recipes(request: RecipeRequest):
    try:
        # Search the database
        query_string = ", ".join(request.ingredients)
        docs = retriever.invoke(query_string)
        
        # Format the recipes for the LLM
        context_list = []
        for doc in docs:
            details = (
                f"RECIPE: {doc.metadata['name']}\n"
                f"TIME: {doc.metadata['time']} mins\n"
                f"SERVINGS: {doc.metadata['servings']}\n"
                f"DIET: {doc.metadata['diet']}\n"
                f"DETAILS: {doc.page_content}\n"
            )
            context_list.append(details)
        
        context = "\n---\n".join(context_list)

        # The Chef's Prompt
        prompt = (
    f"You are a master Chef known for being practical and creative. "
    f"A user wants to cook with: {query_string}.\n"
    f"USER CONSTRAINTS: {request.constraints}\n\n"
    f"TASK:\n"
    f"1. Review the recipes below from our database.\n"
    f"2. If a recipe matches the constraints, suggest it normally.\n"
    f"3. IF NO RECIPE FITS THE CONSTRAINTS (like the {request.constraints}), "
    f"DO NOT REFUSE. Instead, pick the closest recipe and provide a 'Chef's Shortcut' "
    f"to make it work. For example, if they only have 15 mins, tell them how to "
    f"skip steps or use higher heat to finish it faster.\n"
    f"4. Always mention the original Time, Servings, and Diet from the metadata.\n\n"
    f"DATABASE ENTRIES:\n{context}\n\n"
    f"Keep your tone encouraging and solution-oriented!"
)
        
        response = llm.invoke(prompt)
        return {"chef_advice": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

