# Test if all packages are installed correctly
try:
    import langchain
    print("✓ langchain installed")
    
    import faiss
    print("✓ faiss (vector database) installed")
    
    from pypdf import PdfReader
    print("✓ pypdf installed")
    
    import pandas
    print("✓ pandas installed")
    
    import openpyxl
    print("✓ openpyxl installed")
    
    print("\n🎉 Core packages installed successfully!")
    print("\n⚠️  Note: sentence-transformers needs Visual C++ Redistributable")
    print("We'll use OpenAI embeddings instead (you'll need an API key)")
    print("\n✅ Your environment is ready for Step 2!")
    
except ImportError as e:
    print(f"❌ Error: {e}")
