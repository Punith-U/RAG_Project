from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Connect to database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="rag_documents"
)

print("=" * 70)
print("📊 EXPORTING ALL CHUNKS TO EXCEL TABLE")
print("=" * 70)

# Get all documents
print("\n🔄 Retrieving all chunks...")
results = vectorstore.similarity_search("", k=1500)

# Create table data
table_data = []
for idx, doc in enumerate(results, 1):
    table_data.append({
        'Chunk_ID': idx,
        'Source_File': doc.metadata.get('source', 'Unknown'),
        'File_Type': doc.metadata.get('type', 'Unknown').upper(),
        'Content': doc.page_content[:500],  # First 500 characters
        'Full_Length': len(doc.page_content)
    })

# Create DataFrame
df = pd.DataFrame(table_data)

# Save to Excel
excel_file = 'chunks_data.xlsx'
df.to_excel(excel_file, index=False, sheet_name='All_Chunks')

print("\n✅ Export completed!")
print(f"📄 Total chunks: {len(results)}")
print(f"📑 PDF chunks: {len([d for d in results if d.metadata.get('type') == 'pdf'])}")
print(f"📑 Excel chunks: {len([d for d in results if d.metadata.get('type') == 'excel'])}")
print(f"\n💾 Saved to: {excel_file}")

# Print preview
print("\n" + "=" * 70)
print("📋 PREVIEW OF TABLE:")
print("=" * 70)
print(df.head(10).to_string())

print("\n" + "=" * 70)
print("🎉 Your manager can now open chunks_data.xlsx!")
print("=" * 70)
