import os
from loaders.pdf_loader import load_and_process_pdfs, create_vector_store

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()
