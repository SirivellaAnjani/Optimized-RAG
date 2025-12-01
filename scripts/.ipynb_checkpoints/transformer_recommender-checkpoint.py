import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import pandas as pd
import re

'''
Transformer-based movie recommender 
trained on INSPIRED dataset
'''
class TransformerRecommender(nn.Module):

    def __init__(self, model_name="bert-base-uncased", num_movies=20):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Recommendation head
        self.recommender_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_movies)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Get movie scores
        logits = self.recommender_head(cls_output)
        return logits
    
    def predict_top_k(self, conversation_text, movie_id_to_name, k=3):
        """Predict top-k movie recommendations"""
        # Tokenize
        inputs = self.tokenizer(
            conversation_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'], inputs['attention_mask'])
            scores = torch.softmax(logits, dim=-1)
        
        # SAFETY CHECK: Adjust k to available size
        num_available = min(len(movie_id_to_name), scores.shape[1])
        actual_k = min(k, num_available)
        
        # Get top-k movies
        top_k_scores, top_k_indices = torch.topk(scores[0], actual_k)
        
        recommendations = []
        for idx, score in zip(top_k_indices, top_k_scores):
            movie_id = idx.item()
            if movie_id in movie_id_to_name:
                recommendations.append({
                    'movie_id': movie_id,
                    'movie_name': movie_id_to_name[movie_id],
                    'score': score.item()
                })
        
        return recommendations

'''
Process the INSPIRED dataset 
for training the transformer
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
        
        print(f"Loading {len(df)} movies from database...")
        
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
        """
        Get movie ID for a given movie name
        Tries exact match first, then match without year
        """
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

        
    '''
    Load dialog data grouped by conversation
    '''
    '''
    def load_dialogs(self, split="train", max_dialogs=None):
        
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t')
        
        # Group by dialog_id
        dialogs = []
        for dialog_id, group in df.groupby('dialog_id'):
            conversation = []
            recommended_movies = []
            
            for _, row in group.iterrows():
                speaker = row.get('speaker', '')
                utterance = row.get('text', '')
                movie_name = row.get('movies', '')
                
                conversation.append(f"{speaker}: {utterance}")
                
                # Track recommended movies
                if movie_name and movie_name in self.movie_id_map:
                    recommended_movies.append(self.movie_id_map[movie_name])
            
            if conversation and recommended_movies:
                dialogs.append({
                    'dialog_id': dialog_id,
                    'conversation': ' '.join(conversation),
                    'recommended_movies': list(set(recommended_movies))
                })
            
            if max_dialogs and len(dialogs) >= max_dialogs:
                break
        
        print(f"Loaded {len(dialogs)} dialogs")
        return dialogs
    '''
    def load_dialogs(self, split="train", max_dialogs=None):
        
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t')
        
        print(f"\n{'='*60}")
        print(f"Loading dialogs from {split}.tsv")
        print(f"{'='*60}")
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
                    
                    # Use the new method that handles year matching
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
        
        print(f"\n{'='*60}")
        print(f"Loading Summary")
        print(f"{'='*60}")
        print(f"✓ Valid dialogs loaded: {len(dialogs)}")
        print(f"✗ Dialogs skipped (no conversation): {dialogs_skipped_no_conv}")
        print(f"✗ Dialogs skipped (no matching movies): {dialogs_skipped_no_movies}")
        print(f"  Total movie mentions processed: {total_movie_mentions_processed}")
        print(f"  Movies matched to database: {movies_matched}")
        print(f"  Movies NOT in database: {movies_not_in_db}")
        print(f"  Match rate: {movies_matched/total_movie_mentions_processed*100:.1f}%")
        print(f"{'='*60}\n")
        
        return dialogs