from dotenv import load_dotenv
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load image mapping (for future multimodal support)
if os.path.exists("image_mapping.json"):
    with open("image_mapping.json", "r") as f:
        image_mapping = json.load(f)
else:
    image_mapping = []

# Load Qdrant vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="rag_documents"
)

# Initialize OpenAI Chat LLM client
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-3.5-turbo",
    max_tokens=1024
)

def ask_bot(user_question):
    # Search relevant chunks
    results = vectorstore.similarity_search(user_question, k=5)
    context = "\n\n".join([doc.page_content for doc in results])
    
    prompt = (
        f"You are a helpful assistant. Answer the following question based "
        f"only on the context below. If an image is mentioned in the context, mention it.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {user_question}\n\n"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Bot answer:\n", response.content)

if __name__ == "__main__":
    question = input("Ask a question: ")
    ask_bot(question)
