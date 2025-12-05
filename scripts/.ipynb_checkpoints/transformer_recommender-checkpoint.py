import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


'''
Transformer-based recommender for LLM + RAG system
Uses BERT encoder with trained recommender head
Trained on INSPIRED dataset
'''
class TransformerRecommender(nn.Module):

    '''
    Args:
        - model_name: HuggingFace model name for encoder
        - num_movies: Number of movies in the dataset
    '''
    def __init__(self, num_movies, model_name="bert-base-uncased"):
        
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

    """   
    Args:
        - input_ids: Tokenized input (batch_size, seq_len)
        - attention_mask: Attention mask (batch_size, seq_len)
        
    Returns:
        - logits: Movie scores (batch_size, num_movies)
    """
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Get movie scores
        logits = self.recommender_head(cls_output)
        return logits
    
    def predict_top_k(self, conversation_text, movie_id_to_name, k=10):
        
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

