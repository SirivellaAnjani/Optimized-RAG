'''
Data loading utilities for INSPIRED dataset
'''
from pathlib import Path
import csv
import pandas as pd
import re
from llama_index.core import Document
from tqdm import tqdm

'''
Functions:
    - Load INSPIRED dataset from TSV 
    - Convert to documents

Args:
    - data_path: Path to the TSV file
    - max_rows: Maximum number of rows to load (None = load all rows)
'''
def load_inspired_dataset(data_path, max_rows=None):

    if not Path(data_path).exists():
        raise FileNotFoundError(f"INSPIRED dataset not found at '{data_path}'.")
    
    documents = []
    
    # First pass: count total rows for progress bar
    with open(data_path, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1
    
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)
    
    # 2nd pass: Load the TSV data
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for idx, row in enumerate(tqdm(reader, total=total_rows, desc="Loading data")):
            if max_rows is not None and idx >= max_rows:
                break
                
            # Extract conversation information from TSV
            dialog_id = row.get('dialog_id', '')
            turn_id = row.get('turn_id', '')
            utterance = row.get('utterance', '')
            speaker = row.get('speaker', '')
            movie_name = row.get('movie_name', '')
            
            # Create document text
            if speaker == "RECOMMENDER":
                doc_text = f"Recommendation: {utterance}\n"
            else:
                doc_text = f"User preference: {utterance}\n"
            
            if movie_name:
                doc_text += f"Movie mentioned: {movie_name}\n"
            
            # Create metadata
            metadata = {
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "movie_name": movie_name if movie_name else "None"
            }
            
            # Create Document object
            doc = Document(text=doc_text, metadata=metadata)
            documents.append(doc)
    
    print(f"Loaded {len(documents)} turns from INSPIRED dataset")
    return documents

'''
Functions:
    - Load the unprocessed INSPIRED movie database for reference
Args:
    - dataset_dir: Directory of the TSV file
Returns:
    - Dictionary mapping movieId (index) to movie details
'''
def load_movie_database(dataset_dir="data"):
    
    movie_db_path = Path(dataset_dir) / "raw" / "movie_database.tsv"
    
    if not movie_db_path.exists():
        print(f"Movie database not found at {movie_db_path}")
        return {}
    
    df = pd.read_csv(movie_db_path, sep='\t', encoding='utf-8')
    
    print(f"Loading {len(df)} movies from database...")
    
    movie_name_col = 'title' 
    
    # Tracking filtering
    missing_titles = 0
    nan_titles = 0
    valid_movies = 0
    
    movies = {}
    
    for idx, row in df.iterrows():
        movie_name = str(row[movie_name_col])
        
        # Skip invalid entries
        if pd.isna(row[movie_name_col]):
            missing_titles += 1
            continue
            
        if movie_name == 'nan':
            nan_titles += 1
            continue
        
        movies[idx] = row.to_dict()
        valid_movies += 1
    
    print(f"Loaded {valid_movies} valid movies from movie database")
    print(f"Skipped: {missing_titles} missing titles + {nan_titles} 'nan' titles")
    
    return movies


'''
Process the INSPIRED dataset 
'''
class INSPIREDDataProcessor:
    
    def __init__(self, dataset_dir="data"):
        self.dataset_dir = Path(dataset_dir)
        self.movie_id_map = {}
        self.movie_name_map = {}
        self.movie_name_to_id = {}

    @staticmethod
    def strip_year(movie_name):
        
        if pd.isna(movie_name):
            return None
        
        # Remove (year)
        movie_name = str(movie_name).strip()
        movie_name = re.sub(r'\s*\(\d{4}\)\s*$', '', movie_name)
        return movie_name.strip()
    
    '''
    Load movie database and create mappings
    '''
    def load_movie_database(self):
        
        movie_db_path = self.dataset_dir / "raw" / "movie_database.tsv"
        
        if not movie_db_path.exists():
            raise FileNotFoundError(f"Movie database not found at {movie_db_path}")
        
        # Load with pandas
        df = pd.read_csv(movie_db_path, sep='\t', encoding='utf-8')
        
        print(f"In load_movie_database().\nLoading {len(df)} movies from database...")
        
        movie_name_col = 'title'
        
        # Tracking filtering
        missing_titles = 0
        nan_titles = 0
        valid_movies = 0
        
        # Create movie ID mappings
        for idx, row in df.iterrows():
            movie_name = str(row[movie_name_col])
            
            if pd.isna(row[movie_name_col]):
                missing_titles += 1
                continue
                
            if movie_name == 'nan':
                nan_titles += 1
                continue

            # Valid movie
            self.movie_id_map[movie_name] = idx
            self.movie_name_map[idx] = movie_name

            # ALSO store without year for matching
            movie_name_no_year = self.strip_year(movie_name)
            if movie_name_no_year:
                self.movie_name_to_id[movie_name_no_year] = idx
                
            valid_movies += 1
                
        print(f"After Filtering, loaded {valid_movies} movies")
        print(f"Skipped: {missing_titles} missing titles + {nan_titles} 'nan' titles")
        print(f"Total filtered: {missing_titles + nan_titles}")
        
        return self.movie_id_map, self.movie_name_map  

    def get_movie_id(self, movie_name):

        if pd.isna(movie_name) or movie_name == '':
            return None
        
        movie_name = str(movie_name).strip()
        
        # Try exact match first
        if movie_name in self.movie_id_map:
            return self.movie_id_map[movie_name]
        
        # Try without year
        movie_name_no_year = self.strip_year(movie_name)
        if movie_name_no_year in self.movie_name_to_id:
            return self.movie_name_to_id[movie_name_no_year]
        
        return None

    def load_dialogs(self, split="train", max_dialogs=None):
        
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t')
        
        print(f"Loading dialogs from {split}.tsv")        
        print(f"Total turns in file: {len(df)}")
        print(f"Unique dialogs: {df['dialog_id'].nunique()}")
        
        # Count movie mentions in the data
        movie_mentions = df[df['movies'].notna()]['movies']
        print(f"Total movie mentions: {len(movie_mentions)}")
        print(f"Unique movies mentioned: {movie_mentions.nunique()}")
        
        # Group by dialog_id
        dialogs = []
        dialogs_skipped_no_conv = 0
        dialogs_skipped_no_movies = 0
        total_movie_mentions_processed = 0
        movies_matched = 0
        movies_not_in_db = 0
        
        for dialog_id, group in df.groupby('dialog_id'):
            conversation = []
            recommended_movies = []
            
            for _, row in group.iterrows():
                speaker = row.get('speaker', '')
                utterance = row.get('text', '')
                movie_name = row.get('movies', '')
                
                conversation.append(f"{speaker}: {utterance}")
                
                # Track recommended movies
                if movie_name and pd.notna(movie_name) and movie_name != '':
                    total_movie_mentions_processed += 1
                    
                    movie_id = self.get_movie_id(movie_name)
                    
                    if movie_id is not None:
                        recommended_movies.append(movie_id)
                        movies_matched += 1
                    else:
                        movies_not_in_db += 1
            
            if not conversation:
                dialogs_skipped_no_conv += 1
                continue
                
            if not recommended_movies:
                dialogs_skipped_no_movies += 1
                continue
            
            dialogs.append({
                'dialog_id': dialog_id,
                'conversation': ' '.join(conversation),
                'recommended_movies': list(set(recommended_movies))
            })
            
            if max_dialogs and len(dialogs) >= max_dialogs:
                break
        
        
        print(f"Loading Summary")
        print(f"  Valid dialogs loaded: {len(dialogs)}")
        print(f"  Dialogs skipped (no conversation): {dialogs_skipped_no_conv}")
        print(f"  Dialogs skipped (no matching movies): {dialogs_skipped_no_movies}")
        print(f"  Total movie mentions processed: {total_movie_mentions_processed}")
        print(f"  Movies matched to database: {movies_matched}")
        print(f"  Movies NOT in database: {movies_not_in_db}")
        print(f"  Match rate: {movies_matched/total_movie_mentions_processed*100:.1f}%")
        
        return dialogs