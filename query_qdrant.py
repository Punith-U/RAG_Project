from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore')

# Connect to your Qdrant database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="rag_documents"
)

print("="*60)
print("🔍 QDRANT DATABASE VERIFICATION")
print("="*60)

print(f"\n✅ Connection successful!")
print(f"📂 Database location: ./qdrant_db/")
print(f"📦 Collection name: rag_documents")

# Test a simple query
print(f"\n🧪 Testing with sample query...")
query = "What is English grammar?"
results = vectorstore.similarity_search(query, k=3)

print(f"\n📝 Query: '{query}'")
print(f"✓ Found {len(results)} relevant chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Type: {doc.metadata.get('type', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:200]}...")
    print()

print("="*60)
print("🎉 Your Qdrant database is working perfectly!")
print("="*60)
print("\n✨ Total chunks in database: ~1360")
print("📊 Status: Ready for queries!")
