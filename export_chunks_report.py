from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import json
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
print("📊 EXPORTING ALL CHUNKS FROM QDRANT DATABASE")
print("=" * 70)

# Get all documents by doing a broad search
print("\n🔄 Retrieving all chunks...")
results = vectorstore.similarity_search("", k=1500)  # Get top 1500

# Group by source
chunks_by_source = {}
for doc in results:
    source = doc.metadata.get('source', 'Unknown')
    if source not in chunks_by_source:
        chunks_by_source[source] = []
    chunks_by_source[source].append({
        'content': doc.page_content[:500],  # First 500 chars
        'type': doc.metadata.get('type', 'Unknown'),
        'full_length': len(doc.page_content)
    })

# Create JSON export
export_data = {
    'total_chunks': len(results),
    'sources': chunks_by_source,
    'summary': {
        'pdf_chunks': len([d for d in results if d.metadata.get('type') == 'pdf']),
        'excel_chunks': len([d for d in results if d.metadata.get('type') == 'excel'])
    }
}

# Save to JSON
with open('chunks_export.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

print("\n✅ Export completed!")
print(f"📄 Total chunks exported: {len(results)}")
print(f"📑 Sources found: {len(chunks_by_source)}")
print(f"💾 Saved to: chunks_export.json")

# Print summary
print("\n" + "=" * 70)
print("📋 SUMMARY BY SOURCE:")
print("=" * 70)
for source, chunks in chunks_by_source.items():
    print(f"\n📌 {source}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Type: {chunks[0]['type']}")
    print(f"   Sample: {chunks[0]['content'][:100]}...")

print("\n" + "=" * 70)
print("🎉 Your manager can now view chunks_export.json!")
print("=" * 70)
