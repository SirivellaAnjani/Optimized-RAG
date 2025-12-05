'''
RAG functions for building index
'''
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from pathlib import Path
from .data_loader import load_inspired_dataset, load_movie_database


'''
Function:
    - Load INSPIRED dataset
    - Create vector index

Args:
    - dataset_dir: Directory containing the dataset
    - split: Which split to use (train, dev, test)
    - max_rows: Maximum number of rows to load (None for all rows)

Returns:
    - index: VectorStoreIndex for RAG
'''
import shutil

def load_and_index_documents(dataset_dir="data", split="train", max_rows=None, persist_dir="data/index", force_rebuild=False):
    
    persist_path = Path(persist_dir) / split
    
    # Force rebuild if TRUE
    if force_rebuild and persist_path.exists():
        print("Force rebuild requested. Deleting existing index...")
        shutil.rmtree(persist_path)
    
    # Try loading existing index first
    if persist_path.exists():
        print(f"Loading existing index from {persist_path}...")
        
        try:
            from llama_index.core import StorageContext, load_index_from_storage
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully!")
            
            return index
        
        except Exception as e:
            print(f"Failed to load index: {e}")
            print("Rebuilding index...")    

    
    # Construct path to the data file 
    data_path = Path(dataset_dir) / "processed" / f"{split}.tsv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. \nCheck for typos."
        )
    
    # Load INSPIRED documents with max_rows limit
    docs = load_inspired_dataset(data_path, max_rows=max_rows)
    
    if not docs:
        raise ValueError("No documents loaded from INSPIRED dataset")
    
    
    # Build vector index from documents
    print("Building vector index...")
    index = VectorStoreIndex.from_documents(
        docs,         
        show_progress=True
    )

    # Save the index
    print(f"Saving index to {persist_path}...")
    persist_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_path))
    print("Index saved!")
    
    return index