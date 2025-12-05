"""
Baseline recommender for LLM + RAG system
Uses popularity-based recommendations
"""
from collections import defaultdict

class PopularityRecommender:
    """
    Popularity-based baseline recommender
    Recommends most frequently mentioned movies in training data
    """
    
    def __init__(self, data_processor):
        """
        Args:
            data_processor: INSPIREDDataProcessor instance
        """
        self.data_processor = data_processor
        self.popularity_scores = {}
        
    def fit(self, split="train"):
        """
        Calculate movie popularity from training data
        
        Args:
            split: Dataset split to calculate popularity from
        """
        print(f"\nCalculating movie popularity from {split} data...")
        dialogs = self.data_processor.load_dialogs(split=split, max_dialogs=None)
        
        # Count movie occurrences
        movie_counts = defaultdict(int)
        for dialog in dialogs:
            for movie_id in dialog['recommended_movies']:
                movie_counts[movie_id] += 1
        
        # Store as sorted list (most popular first)
        self.popularity_scores = sorted(
            movie_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"Calculated popularity for {len(self.popularity_scores)} movies")
        print(f"Most popular movie ID: {self.popularity_scores[0][0]} (count: {self.popularity_scores[0][1]})")
        
    def predict_top_k(self, conversation_text, movie_id_to_name, k=10):
        """
        Predict top-k most popular movies
        
        Args:
            conversation_text: Not used (popularity is global)
            movie_id_to_name: Dictionary mapping movie IDs to names
            k: Number of recommendations
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        for movie_id, count in self.popularity_scores[:k]:
            if movie_id in movie_id_to_name:
                recommendations.append({
                    'movie_id': movie_id,
                    'movie_name': movie_id_to_name[movie_id],
                    'score': count
                })
        
        return recommendations