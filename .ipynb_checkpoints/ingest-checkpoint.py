import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Define the correct path to your data
# This looks inside the 'data' folder for 'FinalRecipe.csv'
csv_path = os.path.join("data", "FinalRecipe.csv")

print(f"Loading {csv_path}...")

# Check if the file exists before trying to read it to prevent crashes
if not os.path.exists(csv_path):
    print(f"❌ Error: Could not find the file at {csv_path}")
    print("Please make sure 'FinalRecipe.csv' is inside the 'data' folder.")
else:
    df = pd.read_csv(csv_path)
    print("✅ File loaded successfully!")

    documents = []
    for index, row in df.iterrows():
        # Using your specific column names from the image
        content = f"Recipe: {row['RecipeName']}\nIngredients: {row['cleaned_ings']}\nInstructions: {row['Instructions']}"
        
        # Adding your requested metadata: Servings and Cooking Time
        metadata = {
            "name": row['RecipeName'],
            "time": row['TotalTimeInMins'],
            "servings": row['Servings'],
            "diet": row['Diet']
        }
        
        documents.append(Document(page_content=content, metadata=metadata))

    # 2. Create the vectors (Local CPU)
    print("Creating local index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Save to FAISS
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("vector_db")
    print("✅ Success! Your 'vector_db' folder is created and ready.")