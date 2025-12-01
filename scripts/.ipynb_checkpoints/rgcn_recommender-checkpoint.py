import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class RGCNRecommender(nn.Module):
    """RGCN-based movie recommender"""
    
    def __init__(self, num_nodes, num_relations, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embeddings (users and movies)
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        
        # Relation-specific weight matrices for each layer
        self.relation_weights_layer1 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for _ in range(num_relations)
        ])
        
        self.relation_weights_layer2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) 
            for _ in range(num_relations)
        ])
        
        # Self-loop weights
        self.self_weight_layer1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.self_weight_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Output layer for scoring
        self.score_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def aggregate_neighbors(self, x, edge_index, edge_type, relation_weights):
        """
        Manual neighbor aggregation for RGCN
        
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge connections [2, num_edges]
            edge_type: Edge types [num_edges]
            relation_weights: ModuleList of relation-specific transformations
        
        Returns:
            Aggregated embeddings [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        output = torch.zeros_like(x)
        
        # Group edges by relation type
        for rel_type in range(self.num_relations):
            # Get edges of this type
            mask = (edge_type == rel_type)
            if mask.sum() == 0:
                continue
            
            rel_edges = edge_index[:, mask]
            src_nodes = rel_edges[0]  # Source nodes
            dst_nodes = rel_edges[1]  # Destination nodes
            
            # Get source node embeddings
            src_embs = x[src_nodes]
            
            # Apply relation-specific transformation
            transformed = relation_weights[rel_type](src_embs)
            
            # Aggregate to destination nodes (sum aggregation)
            output.index_add_(0, dst_nodes, transformed)
        
        return output
        
    def forward(self, edge_index, edge_type, user_idx, movie_indices):
        """
        Forward pass through RGCN
        
        Args:
            edge_index: Edge connections [2, num_edges]
            edge_type: Edge types [num_edges]
            user_idx: User node index (single value)
            movie_indices: Movie node indices to score [num_movies]
        
        Returns:
            scores: Predicted scores [num_movies]
            x: Final node embeddings [num_nodes, hidden_dim]
        """
        # Get initial embeddings
        x = self.node_embeddings.weight
        
        # Layer 1: RGCN aggregation
        h = self.aggregate_neighbors(x, edge_index, edge_type, self.relation_weights_layer1)
        h = h + self.self_weight_layer1(x)  # Add self-loop
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        
        # Layer 2: RGCN aggregation
        h = self.aggregate_neighbors(h, edge_index, edge_type, self.relation_weights_layer2)
        h = h + self.self_weight_layer2(h)  # Add self-loop
        
        # Get user and movie embeddings
        user_emb = h[user_idx].unsqueeze(0).repeat(len(movie_indices), 1)
        movie_embs = h[movie_indices]
        
        # Concatenate and score
        combined = torch.cat([user_emb, movie_embs], dim=1)
        scores = self.score_predictor(combined).squeeze()
        
        return scores, h
    
    def predict_top_k(self, edge_index, edge_type, user_idx, candidate_movies, k=10):
        """
        Predict top-k movie recommendations for a user
        
        Args:
            edge_index: Edge connections [2, num_edges]
            edge_type: Edge types [num_edges]
            user_idx: User node index
            candidate_movies: Candidate movie indices [num_candidates]
            k: Number of recommendations
        
        Returns:
            List of dicts with 'movie_node_idx' and 'score'
        """
        self.eval()
        
        with torch.no_grad():
            scores, _ = self.forward(edge_index, edge_type, user_idx, candidate_movies)
            
            # Handle single score case
            if len(scores.shape) == 0:
                scores = scores.unsqueeze(0)
            
            # Get top-k
            actual_k = min(k, len(scores))
            top_k_scores, top_k_indices = torch.topk(scores, actual_k)
            
            recommendations = []
            for idx, score in zip(top_k_indices, top_k_scores):
                movie_node_idx = candidate_movies[idx.item()]
                recommendations.append({
                    'movie_node_idx': movie_node_idx.item(),
                    'score': score.item()
                })
        
        return recommendations

class INSPIREDGraphBuilder:
    """Build knowledge graph from INSPIRED dataset"""
    
    def __init__(self, dataset_dir="inspired_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.node_counter = 0
        
        # Relation types
        self.relation_types = {
            'USER_LIKES': 0,
            'USER_DISLIKES': 1,
            'USER_MENTIONED': 2,
            'MOVIE_GENRE': 3,
            'MOVIE_SIMILAR': 4
        }
        
        self.edges = defaultdict(list)

    def match_movie_name(self, dialog_movie):
        """Try to match a movie name from dialogs to database movies"""
        
        if not dialog_movie or pd.isna(dialog_movie):
            return None
        
        # Try exact match
        if dialog_movie in self.movie_to_idx:
            return self.movie_to_idx[dialog_movie]
        
        # Try removing year (e.g., "Knives Out (2019)" -> "Knives Out")
        import re
        movie_without_year = re.sub(r'\s*\(\d{4}\)\s*$', '', dialog_movie).strip()
        if movie_without_year in self.movie_to_idx:
            return self.movie_to_idx[movie_without_year]
        
        # Try case-insensitive exact match
        dialog_lower = dialog_movie.lower()
        for db_movie, idx in self.movie_to_idx.items():
            if db_movie.lower() == dialog_lower:
                return idx
        
        # Try case-insensitive match without year
        movie_without_year_lower = movie_without_year.lower()
        for db_movie, idx in self.movie_to_idx.items():
            if db_movie.lower() == movie_without_year_lower:
                return idx
        
        return None
    
    def add_node(self, node_name, node_type='movie'):
        """Add a node and return its index"""
        if node_type == 'movie':
            if node_name not in self.movie_to_idx:
                self.movie_to_idx[node_name] = self.node_counter
                self.idx_to_movie[self.node_counter] = node_name
                self.node_counter += 1
            return self.movie_to_idx[node_name]
        else:  # user
            if node_name not in self.user_to_idx:
                self.user_to_idx[node_name] = self.node_counter
                self.idx_to_user[self.node_counter] = node_name
                self.node_counter += 1
            return self.user_to_idx[node_name]
    
    def add_edge(self, src_idx, dst_idx, relation_type):
        """Add an edge to the graph"""
        self.edges[relation_type].append([src_idx, dst_idx])
    


    def load_movie_database(self):
        """Load movie database with proper name matching"""
        movie_db_path = self.dataset_dir / "raw" / "movie_database.tsv"
        
        if not movie_db_path.exists():
            raise FileNotFoundError(f"Movie database not found at {movie_db_path}")
        
        df = pd.read_csv(movie_db_path, sep='\t')
        
        print(f"Loading {len(df)} movies into graph...")
        
        # Add all movies as nodes WITH YEAR STRIPPED
        for idx, row in df.iterrows():
            movie_name_with_year = row.get('title', 'Unknown')
            
            # Store with year (original)
            self.add_node(movie_name_with_year, node_type='movie')
            
            # ALSO store without year for matching
            import re
            movie_name_without_year = re.sub(r'\s*\(\d{4}\)\s*$', '', str(movie_name_with_year)).strip()
            if movie_name_without_year != movie_name_with_year:
                # Create alias mapping
                if not hasattr(self, 'movie_aliases'):
                    self.movie_aliases = {}
                self.movie_aliases[movie_name_without_year] = movie_name_with_year
        
        print(f"Added {len(self.movie_to_idx)} movie nodes")
        return df
    
    
    def build_graph_from_dialogs(self, split="train", max_dialogs=None):
        """Build graph from dialog data"""
        dialog_path = self.dataset_dir / "processed" / f"{split}.tsv"
        
        if not dialog_path.exists():
            raise FileNotFoundError(f"Dialog file not found at {dialog_path}")
        
        df = pd.read_csv(dialog_path, sep='\t')
        
        print(f"\nBuilding graph from dialogs...")
        print(f"Using 'movies' and 'text' columns")
        
        # Count matches
        matched_count = 0
        unmatched_count = 0
        dialog_count = 0
        edges_added = 0
        
        for dialog_id, group in tqdm(df.groupby('dialog_id'), desc="Processing dialogs"):
            user_idx = self.add_node(str(dialog_id), node_type='user')
            
            for _, row in group.iterrows():
                # Get movie name from 'movies' column
                movie_name = row.get('movies', '')
                utterance = str(row.get('text', '')).lower()
                
                # Skip if no movie mentioned
                if pd.isna(movie_name) or movie_name == '':
                    continue
                
                # Handle if movie_name contains multiple movies (semicolon or comma-separated)
                if isinstance(movie_name, str):
                    if ';' in movie_name:
                        movie_names = [m.strip() for m in movie_name.split(';')]
                    elif ',' in movie_name:
                        movie_names = [m.strip() for m in movie_name.split(',')]
                    else:
                        movie_names = [movie_name.strip()]
                else:
                    movie_names = [movie_name]
                
                for single_movie in movie_names:
                    if not single_movie or pd.isna(single_movie):
                        continue
                    
                    movie_idx = None
                    
                    # Strategy 1: Exact match
                    if single_movie in self.movie_to_idx:
                        movie_idx = self.movie_to_idx[single_movie]
                        matched_count += 1
                    else:
                        # Strategy 2: Strip year and try again
                        import re
                        single_movie_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', single_movie).strip()
                        
                        # Check if we have an alias mapping
                        if hasattr(self, 'movie_aliases') and single_movie_no_year in self.movie_aliases:
                            actual_name = self.movie_aliases[single_movie_no_year]
                            if actual_name in self.movie_to_idx:
                                movie_idx = self.movie_to_idx[actual_name]
                                matched_count += 1
                        
                        # Strategy 3: Try the no-year version directly
                        if movie_idx is None and single_movie_no_year in self.movie_to_idx:
                            movie_idx = self.movie_to_idx[single_movie_no_year]
                            matched_count += 1
                        
                        # Strategy 4: Case-insensitive match
                        if movie_idx is None:
                            single_lower = single_movie.lower()
                            single_no_year_lower = single_movie_no_year.lower()
                            
                            for db_movie, idx in self.movie_to_idx.items():
                                db_lower = db_movie.lower()
                                db_no_year = re.sub(r'\s*\(\d{4}\)\s*$', '', db_movie).strip().lower()
                                
                                if db_lower == single_lower or db_no_year == single_no_year_lower:
                                    movie_idx = idx
                                    matched_count += 1
                                    break
                    
                    if movie_idx is None:
                        unmatched_count += 1
                        continue
                    
                    # Determine relation type based on sentiment in utterance
                    if any(word in utterance for word in ['love', 'like', 'enjoy', 'favorite', 'great', 'amazing', 'excellent', 'best', 'recommend']):
                        relation = self.relation_types['USER_LIKES']
                    elif any(word in utterance for word in ['hate', 'dislike', 'boring', 'bad', 'terrible', 'worst', 'awful', 'didn\'t like']):
                        relation = self.relation_types['USER_DISLIKES']
                    else:
                        relation = self.relation_types['USER_MENTIONED']
                    
                    self.add_edge(user_idx, movie_idx, relation)
                    edges_added += 1
            
            dialog_count += 1
            if max_dialogs and dialog_count >= max_dialogs:
                break
        
        print(f"Processed {dialog_count} conversations")
        print(f"Total nodes: {self.node_counter}")
        print(f"Total users: {len(self.user_to_idx)}")
        print(f"Total movies: {len(self.movie_to_idx)}")
        print(f"Edges added: {edges_added}")
        print(f"Matched movies: {matched_count}")
        print(f"Unmatched movies: {unmatched_count}")
        
        if edges_added == 0:
            print("\nWARNING: No edges created!")
            print("Checking sample data...")
            sample_movies = df['movies'].dropna().head(20)
            print(f"Sample movies from dialogs: {sample_movies.tolist()}")
            print(f"\nSample movies from database: {list(self.movie_to_idx.keys())[:20]}")
        
        return self.create_graph_data()    
    
    def create_graph_data(self):
        """Create graph data as dictionary"""
        edge_index_list = []
        edge_type_list = []
        
        for rel_type, edge_list in self.edges.items():
            if edge_list:
                edge_array = np.array(edge_list).T
                edge_index_list.append(edge_array)
                edge_type_list.extend([rel_type] * edge_array.shape[1])
        
        if not edge_index_list:
            # Print debug info
            print("\nERROR: No edges found!")
            print(f"Total edge types checked: {len(self.edges)}")
            for rel_type, edges in self.edges.items():
                print(f"  Relation {rel_type}: {len(edges)} edges")
            raise ValueError("No edges in graph!")
        
        edge_index = np.concatenate(edge_index_list, axis=1)
        edge_type = np.array(edge_type_list)
        
        graph_data = {
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_type': torch.tensor(edge_type, dtype=torch.long),
            'num_nodes': self.node_counter
        }
        
        print(f"Created graph with {graph_data['num_nodes']} nodes and {graph_data['edge_index'].shape[1]} edges")
        
        return graph_data



def train_rgcn(model, graph_data, graph_builder, num_epochs=10, lr=0.01):
    """Train RGCN with detailed logging"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    print("TRAINING RGCN MODEL (DEBUG MODE)")
    print("="*60)
    
    history = {'train_loss': [], 'epoch': []}
    
    edge_index = graph_data['edge_index']
    edge_type = graph_data['edge_type']
    
    users = list(graph_builder.user_to_idx.values())
    all_movies = set(graph_builder.movie_to_idx.values())
    
    print(f"Total users: {len(users)}")
    print(f"Total movies: {len(all_movies)}")
    print("="*60)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        num_batches = 0
        skipped_users = 0
        
        # Sample users
        num_users_per_epoch = min(200, len(users))
        sampled_users = np.random.choice(users, size=num_users_per_epoch, replace=False)
        
        for user_idx in sampled_users:
            # Get user's movies
            user_movies = []
            for rel_type, edges in graph_builder.edges.items():
                user_movies.extend([dst for src, dst in edges if src == user_idx])
            
            if not user_movies:
                skipped_users += 1
                continue
            
            user_movies_set = set(user_movies)
            pos_movies_list = list(user_movies_set)[:5]
            
            # Negative sampling
            negative_candidates = list(all_movies - user_movies_set)
            
            if len(negative_candidates) < 5:
                skipped_users += 1
                continue
            
            neg_movies_list = np.random.choice(negative_candidates, size=5, replace=False)
            
            pos_movies = torch.tensor(pos_movies_list, dtype=torch.long)
            neg_movies = torch.tensor(neg_movies_list, dtype=torch.long)
            
            all_movie_samples = torch.cat([pos_movies, neg_movies])
            labels = torch.cat([
                torch.ones(len(pos_movies)),
                torch.zeros(len(neg_movies))
            ])
            
            # Forward
            scores, _ = model(edge_index, edge_type, user_idx, all_movie_samples)
            
            if len(scores.shape) == 0:
                scores = scores.unsqueeze(0)
            
            loss = criterion(scores, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            num_batches += 1
        
        avg_loss = sum(epoch_losses) / max(num_batches, 1)
        
        # Track history
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        # Detailed logging
        if num_batches > 0:
            min_loss = min(epoch_losses)
            max_loss = max(epoch_losses)
            std_loss = np.std(epoch_losses)
            
            print(f"Epoch {epoch+1:2d} | Avg: {avg_loss:.4f} | Min: {min_loss:.4f} | "
                  f"Max: {max_loss:.4f} | Std: {std_loss:.4f} | "
                  f"Batches: {num_batches} | Skipped: {skipped_users}")
        else:
            print(f"Epoch {epoch+1:2d} | NO VALID BATCHES!")
    
    print("="*60)
    print("Training complete!")
    return model, history

''' DURING DEBUGGING
def train_rgcn(model, graph_data, graph_builder, num_epochs=10, lr=0.01):
    """Train the RGCN model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    print("TRAINING RGCN MODEL")
    
    # Initialize history tracking
    history = {
        'train_loss': [],
        'epoch': []
    }
    
    # Extract edge_index and edge_type from dictionary
    edge_index = graph_data['edge_index']
    edge_type = graph_data['edge_type']
    
    # Get all users and movies
    users = list(graph_builder.user_to_idx.values())
    all_movies = set(graph_builder.movie_to_idx.values())
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Sample users for this epoch
        num_users_per_epoch = min(200, len(users))
        sampled_users = np.random.choice(users, size=num_users_per_epoch, replace=False)
        
        for user_idx in sampled_users:
            # Get positive samples (movies user interacted with)
            user_movies = []
            for rel_type, edges in graph_builder.edges.items():
                user_movies.extend([dst for src, dst in edges if src == user_idx])
            
            if not user_movies or len(user_movies) == 0:
                continue
            
            # Deduplicate and limit positive samples
            user_movies_set = set(user_movies)
            pos_movies_list = list(user_movies_set)[:5]
            
            # FIXED: Sample negative movies that user DIDN'T interact with
            negative_candidates = list(all_movies - user_movies_set)
            
            if len(negative_candidates) < 5:
                continue  # Skip if not enough negative samples
            
            neg_movies_list = np.random.choice(negative_candidates, size=5, replace=False)
            
            # Create tensors
            pos_movies = torch.tensor(pos_movies_list, dtype=torch.long)
            neg_movies = torch.tensor(neg_movies_list, dtype=torch.long)
            
            all_movie_samples = torch.cat([pos_movies, neg_movies])
            labels = torch.cat([
                torch.ones(len(pos_movies)),
                torch.zeros(len(neg_movies))
            ])
            
            # Forward pass
            scores, _ = model(edge_index, edge_type, user_idx, all_movie_samples)
            
            if len(scores.shape) == 0:
                scores = scores.unsqueeze(0)
            
            # Compute loss
            loss = criterion(scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Track history
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} ({num_batches} batches)")
    
    print("Training complete!")
    return model, history
    '''