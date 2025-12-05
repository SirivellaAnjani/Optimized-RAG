import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

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


class BaselineEvaluationPipeline:
    """
    Complete evaluation pipeline for Baseline (LLM + RAG with Popularity recommender)
    """
    
    def __init__(self, recommender, data_processor, k_values=[1, 3, 5, 10]):
        """
        Args:
            recommender: PopularityRecommender instance
            data_processor: INSPIREDDataProcessor instance
            k_values: List of k values for evaluation
        """
        self.recommender = recommender
        self.data_processor = data_processor
        self.evaluator = RecommenderEvaluator(k_values=k_values)
        self.predictions = []
        
    def run_evaluation(self, split="test", max_samples=None, top_k=10):
        """
        Run complete evaluation pipeline
        
        Args:
            split: Which split to evaluate (test/dev)
            max_samples: Maximum number of samples to evaluate
            top_k: Number of recommendations to generate per query
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"RUNNING EVALUATION: Baseline (LLM + RAG + Popularity)")
        print(f"{'='*60}\n")
        
        # Load test dialogs
        print(f"Loading {split} dialogs...")
        test_dialogs = self.data_processor.load_dialogs(
            split=split, 
            max_dialogs=max_samples
        )
        
        print(f"Loaded {len(test_dialogs)} dialogs\n")
        
        # Generate predictions
        self.predictions = self._generate_predictions(test_dialogs, top_k)
        
        # Evaluate
        print("\nComputing metrics...")
        results = self.evaluator.evaluate_batch(self.predictions)
        
        # Display results
        self.evaluator.print_results(
            dataset_name=split,
            num_samples=len(test_dialogs)
        )
        
        return results
    
    def _generate_predictions(self, dialogs, top_k):
        """Generate predictions for all dialogs"""
        predictions = []
        
        print("Generating predictions...")
        for dialog in tqdm(dialogs, desc="Processing dialogs"):
            conversation_text = dialog['conversation']
            ground_truth_ids = set(dialog['recommended_movies'])
            
            # Get recommendations
            recommendations = self.recommender.predict_top_k(
                conversation_text,
                self.data_processor.movie_name_map,
                k=top_k
            )
            
            # Extract movie IDs
            recommended_ids = [rec['movie_id'] for rec in recommendations]
            
            predictions.append({
                'recommended': recommended_ids,
                'ground_truth': ground_truth_ids,
                'dialog_id': dialog['dialog_id']
            })
        
        return predictions
    
    def save_results(self, output_path, metadata=None):
        """Save evaluation results"""
        self.evaluator.save_results(
            output_path=output_path,
            model_name="Baseline (LLM+RAG+Popularity)",
            metadata=metadata
        )
    
    def get_results_table(self):
        """Get results as DataFrame"""
        return self.evaluator.get_summary_table()


class RecommenderEvaluator:
    """
    Evaluator for recommendation metrics
    Supports: HIT@K, MRR@K, NDCG@K, Recall@K
    """
    
    def __init__(self, k_values=[1, 3, 5, 10]):
        """
        Args:
            k_values: List of k values to evaluate at
        """
        self.k_values = k_values
        self.results = None
        
    def hit_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        """HIT@K: 1 if any recommended item in top-K is in ground truth, else 0"""
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & ground_truth) > 0 else 0.0
    
    def mrr_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        """MRR@K: Reciprocal of rank of first relevant item"""
        for rank, movie_id in enumerate(recommended[:k], start=1):
            if movie_id in ground_truth:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain"""
        # DCG
        dcg = 0.0
        for rank, movie_id in enumerate(recommended[:k], start=1):
            if movie_id in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG
        idcg = 0.0
        for rank in range(1, min(len(ground_truth), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def recall_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        """Recall@K: Proportion of relevant items found in top-K"""
        top_k = set(recommended[:k])
        hits = len(top_k & ground_truth)
        return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    def evaluate_single(self, recommended: List[int], ground_truth: Set[int]) -> Dict:
        """Evaluate a single recommendation"""
        results = {}
        
        for k in self.k_values:
            results[f'HIT@{k}'] = self.hit_at_k(recommended, ground_truth, k)
            results[f'MRR@{k}'] = self.mrr_at_k(recommended, ground_truth, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(recommended, ground_truth, k)
            results[f'Recall@{k}'] = self.recall_at_k(recommended, ground_truth, k)
        
        return results
    
    def evaluate_batch(self, predictions: List[Dict]) -> Dict:
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: List of dicts with 'recommended' and 'ground_truth' keys
        
        Returns:
            Dictionary with averaged metrics
        """
        all_results = defaultdict(list)
        
        for pred in predictions:
            single_results = self.evaluate_single(
                pred['recommended'], 
                pred['ground_truth']
            )
            for metric, value in single_results.items():
                all_results[metric].append(value)
        
        # Compute averages
        averaged_results = {
            metric: np.mean(values) 
            for metric, values in all_results.items()
        }
        
        # Store results
        self.results = averaged_results
        
        return averaged_results
    
    def print_results(self, dataset_name: str = "", num_samples: int = 0):
        """Pretty print evaluation results"""
        if self.results is None:
            print("No results to display. Run evaluation first.")
            return
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        
        if dataset_name:
            print(f"Dataset: {dataset_name}")
        if num_samples:
            print(f"Number of samples: {num_samples}")
        print()
        
        # Group by metric type
        for metric_type in ['HIT', 'MRR', 'NDCG', 'Recall']:
            print(f"{metric_type}:")
            for k in self.k_values:
                metric_name = f'{metric_type}@{k}'
                if metric_name in self.results:
                    print(f"  @{k:2d}: {self.results[metric_name]:.4f}")
            print()
    
    def save_results(self, output_path: str, model_name: str = "Transformer+LLM+RAG", 
                     metadata: Dict = None):
        """Save evaluation results to JSON"""
        if self.results is None:
            print("No results to save. Run evaluation first.")
            return
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'k_values': self.k_values,
            'metrics': self.results
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get results as pandas DataFrame for easy viewing"""
        if self.results is None:
            return None
        
        # Organize by metric type and k
        data = []
        for metric_type in ['HIT', 'MRR', 'NDCG', 'Recall']:
            row = {'Metric': metric_type}
            for k in self.k_values:
                metric_name = f'{metric_type}@{k}'
                if metric_name in self.results:
                    row[f'@{k}'] = self.results[metric_name]
            data.append(row)
        
        return pd.DataFrame(data)


class TransformerEvaluationPipeline:
    """
    Complete evaluation pipeline for Transformer + LLM + RAG model
    """
    
    def __init__(self, transformer_model, data_processor, k_values=[1, 3, 5, 10]):
        """
        Args:
            transformer_model: Trained TransformerRecommender instance
            data_processor: INSPIREDDataProcessor instance
            k_values: List of k values for evaluation
        """
        self.model = transformer_model
        self.data_processor = data_processor
        self.evaluator = RecommenderEvaluator(k_values=k_values)
        self.predictions = []
        
    def run_evaluation(self, dataset_dir: str = "data", split: str = "test", 
                       max_samples: int = None, top_k: int = 10) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            dataset_dir: Directory containing dataset
            split: Which split to evaluate (test/dev)
            max_samples: Maximum number of samples to evaluate
            top_k: Number of recommendations to generate per query
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"RUNNING EVALUATION: Transformer + LLM + RAG")
        print(f"{'='*60}\n")
        
        # Load test dialogs
        print(f"Loading {split} dialogs...")
        test_dialogs = self.data_processor.load_dialogs(
            split=split, 
            max_dialogs=max_samples
        )
        
        print(f"Loaded {len(test_dialogs)} dialogs\n")
        
        # Generate predictions
        self.predictions = self._generate_predictions(test_dialogs, top_k)
        
        # Evaluate
        print("\nComputing metrics...")
        results = self.evaluator.evaluate_batch(self.predictions)
        
        # Display results
        self.evaluator.print_results(
            dataset_name=split,
            num_samples=len(test_dialogs)
        )
        
        return results
    
    def _generate_predictions(self, dialogs: List[Dict], top_k: int) -> List[Dict]:
        """Generate predictions for all dialogs"""
        predictions = []
        
        print("Generating predictions...")
        for dialog in tqdm(dialogs, desc="Processing dialogs"):
            conversation_text = dialog['conversation']
            ground_truth_ids = set(dialog['recommended_movies'])
            
            # Get recommendations from model
            recommendations = self.model.predict_top_k(
                conversation_text,
                self.data_processor.movie_name_map,
                k=top_k
            )
            
            # Extract movie IDs
            recommended_ids = [rec['movie_id'] for rec in recommendations]
            
            predictions.append({
                'recommended': recommended_ids,
                'ground_truth': ground_truth_ids,
                'dialog_id': dialog['dialog_id']
            })
        
        return predictions
    
    def save_results(self, output_path: str, metadata: Dict = None):
        """Save evaluation results"""
        self.evaluator.save_results(
            output_path=output_path,
            model_name="Transformer+LLM+RAG",
            metadata=metadata
        )
    
    def get_results_table(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        return self.evaluator.get_summary_table()
    
    def analyze_failures(self, top_n: int = 10):
        """
        Analyze cases where model performed poorly
        
        Args:
            top_n: Number of worst cases to analyze
        """
        if not self.predictions:
            print("No predictions available. Run evaluation first.")
            return
        
        # Calculate per-sample NDCG@5 for ranking
        scores = []
        for pred in self.predictions:
            ndcg = self.evaluator.ndcg_at_k(
                pred['recommended'], 
                pred['ground_truth'], 
                k=5
            )
            scores.append({
                'dialog_id': pred['dialog_id'],
                'ndcg@5': ndcg,
                'ground_truth_size': len(pred['ground_truth']),
                'recommended': pred['recommended'][:5]
            })
        
        # Sort by score (ascending = worst first)
        scores.sort(key=lambda x: x['ndcg@5'])
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} WORST PERFORMING CASES")
        print(f"{'='*60}\n")
        
        for i, case in enumerate(scores[:top_n], 1):
            print(f"{i}. Dialog ID: {case['dialog_id']}")
            print(f"   NDCG@5: {case['ndcg@5']:.4f}")
            print(f"   Ground truth items: {case['ground_truth_size']}")
            print(f"   Top-5 recommended IDs: {case['recommended']}")
            print()