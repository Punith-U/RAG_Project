import os
from pypdf import PdfReader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# ADD THIS
from extract_images_from_pdf import extract_images_from_pdf, save_image_mapping

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
        images_all = []

        if not os.path.exists(self.documents_folder):
            print(f"❌ '{self.documents_folder}' not found!")
            return documents
        
        files = os.listdir(self.documents_folder)
        print(f"🔍 Found {len(files)} file(s)\n")
        
        for filename in files:
            file_path = os.path.join(self.documents_folder, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    # Extract images from this PDF
                    images = extract_images_from_pdf(file_path)
                    images_all.extend(images)
                    
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
        # Save extracted image references for later use
        save_image_mapping(images_all)
        return documents, images_all # Note: returns images as well

    def create_qdrant_collection(self, documents):
        print(f"\n📝 Splitting into chunks...")
        texts = self.text_splitter.split_documents(documents)
        print(f"   ✓ {len(texts)} chunks\n")
        # Optionally, for each chunk, you could attach image info if desired (based on metadata)
        print(f"🧠 Creating Qdrant database...")
        print(f"   (This may take 1-2 minutes...)")
        
        vectorstore = Qdrant.from_documents(
            texts,
            self.embeddings,
            path="./qdrant_db",
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
    documents, images = loader.load_all_documents()
    
    if not documents:
        print("\n❌ No documents loaded")
    else:
        print(f"\n✅ Loaded {len(documents)} document(s)")
        print(f"✅ Extracted {len(images)} images from your PDFs (saved to /images)")
        vectorstore = loader.create_qdrant_collection(documents)
        
        print("\n"+"="*60)
        print("🎉 SUCCESS! Documents stored in Qdrant!")
        print("="*60)
        print("\n📂 Qdrant data saved in: qdrant_db/")
        print("🖼️ Image references in: image_mapping.json and images/")
        print("\n💡 Qdrant stores data in a structured folder")
        print("   with collection metadata and chunks!")
