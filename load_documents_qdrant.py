import os
from pypdf import PdfReader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


class DocumentLoaderQdrant:
    def __init__(self, documents_folder="document"):
        self.documents_folder = documents_folder
        
        print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✓ Model loaded!\n")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.collection_name = "rag_documents"
        
    def load_pdf(self, file_path):
        print(f"📄 Loading PDF: {file_path}")
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            text += page.extract_text()
            print(f"   ✓ Page {page_num}")
        return text
    
    def load_excel(self, file_path):
        print(f"📊 Loading Excel: {file_path}")
        df = pd.read_excel(file_path)
        text = df.to_string()
        print(f"   ✓ {len(df)} rows")
        return text
    
    def load_all_documents(self):
        documents = []
        if not os.path.exists(self.documents_folder):
            print(f"❌ '{self.documents_folder}' not found!")
            return documents
        
        files = os.listdir(self.documents_folder)
        print(f"🔍 Found {len(files)} file(s)\n")
        
        for filename in files:
            file_path = os.path.join(self.documents_folder, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    text = self.load_pdf(file_path)
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "type": "pdf"}
                    ))
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    text = self.load_excel(file_path)
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "type": "excel"}
                    ))
            except Exception as e:
                print(f"❌ Error: {filename}: {str(e)}")
        return documents
    
    def create_qdrant_collection(self, documents):
        print(f"\n📝 Splitting into chunks...")
        texts = self.text_splitter.split_documents(documents)
        print(f"   ✓ {len(texts)} chunks\n")
        
        print(f"🧠 Creating Qdrant database...")
        print(f"   (This may take 1-2 minutes...)")
        
        # Create Qdrant vector store - USING NEW FOLDER NAME
        vectorstore = Qdrant.from_documents(
            texts,
            self.embeddings,
            path="./qdrant_db",  # ← CHANGED FROM qdrant_storage
            collection_name=self.collection_name,
        )
        
        print(f"   ✓ Qdrant database created!")
        print(f"   ✓ {len(texts)} chunks stored")
        
        return vectorstore


if __name__ == "__main__":
    print("="*60)
    print("🚀 RAG CHATBOT - QDRANT DOCUMENT LOADER")
    print("="*60)
    
    loader = DocumentLoaderQdrant()
    documents = loader.load_all_documents()
    
    if not documents:
        print("\n❌ No documents loaded")
    else:
        print(f"\n✅ Loaded {len(documents)} document(s)")
        vectorstore = loader.create_qdrant_collection(documents)
        
        print("\n"+"="*60)
        print("🎉 SUCCESS! Documents stored in Qdrant!")
        print("="*60)
        print("\n📂 Qdrant data saved in: qdrant_db/")  # ← CHANGED
        print("✨ Your manager can see the chunks in Qdrant!")
        print("\n💡 Tip: Qdrant stores data in a structured folder")
        print("   with collection metadata and chunks!")
