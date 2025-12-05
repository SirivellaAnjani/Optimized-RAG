import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from .baseline import PopularityRecommender



class RecommenderEvaluator:
    
    def __init__(self, k_values=[1, 3, 5, 10]):
        self.k_values = k_values
        self.results = None
        
    def hit_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & ground_truth) > 0 else 0.0
    
    def mrr_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
        for rank, movie_id in enumerate(recommended[:k], start=1):
            if movie_id in ground_truth:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, recommended: List[int], ground_truth: Set[int], k: int) -> float:
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
        top_k = set(recommended[:k])
        hits = len(top_k & ground_truth)
        return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    def evaluate_single(self, recommended: List[int], ground_truth: Set[int]) -> Dict:
        results = {}
        
        for k in self.k_values:
            results[f'HIT@{k}'] = self.hit_at_k(recommended, ground_truth, k)
            results[f'MRR@{k}'] = self.mrr_at_k(recommended, ground_truth, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(recommended, ground_truth, k)
            results[f'Recall@{k}'] = self.recall_at_k(recommended, ground_truth, k)
        
        return results
    
    def evaluate_batch(self, predictions: List[Dict]) -> Dict:

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
        if self.results is None:
            print("No results to display. Run evaluation first.")
            return
    
        print("EVALUATION RESULTS")
        
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
    
    def save_results(self, output_path: str, model_name: str, metadata: Dict = None):

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


class EvaluationPipeline:

    def __init__(self, model, data_processor, model_name, k_values=[1, 3, 5, 10]):
        self.model = model
        self.data_processor = data_processor
        self.model_name = model_name
        self.evaluator = RecommenderEvaluator(k_values=k_values)
        self.predictions = []
        
    def run_evaluation(self, split="test", max_samples=None, top_k=10):
        
        print(f"RUNNING EVALUATION: {self.model_name}")
        
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
        
        predictions = []
        
        print("Generating predictions...")
        for dialog in tqdm(dialogs, desc="Processing dialogs"):
            conversation_text = dialog['conversation']
            ground_truth_ids = set(dialog['recommended_movies'])
            
            # Get recommendations
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
    
    def save_results(self, output_path, metadata=None):

        self.evaluator.save_results(
            output_path=output_path,
            model_name=self.model_name,
            metadata=metadata
        )
    
    def get_results_table(self):

        return self.evaluator.get_summary_table()
    
    def analyze_failures(self, top_n=10):
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
        
        print(f"TOP {top_n} WORST PERFORMING CASES")
        
        for i, case in enumerate(scores[:top_n], 1):
            print(f"{i}. Dialog ID: {case['dialog_id']}")
            print(f"   NDCG@5: {case['ndcg@5']:.4f}")
            print(f"   Ground truth items: {case['ground_truth_size']}")
            print(f"   Top-5 recommended IDs: {case['recommended']}")
            print()



class ContextualEvaluator:
    
    def __init__(self, queries_dir="data/evaluation"):
      
        self.queries_dir = Path(queries_dir)
        self.queries = {}
        self.responses = {}
        self.results_file = self.queries_dir / "contextual_results.json"
        
    def load_queries(self):
        
        query_files = [
            "temporal_shift_queries.json",
            "mood_context_queries.json",
            "audience_context_queries.json"
        ]
        
        for filename in query_files:
            filepath = self.queries_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.queries.update(json.load(f))
        
        print(f"Loaded {len(self.queries)} contextual queries")
        return self.queries
    
    def run_queries(self, chat_engine):
        
        print("\nRunning contextual queries...")
        
        self.responses = {}
        
        for query_id, query_data in tqdm(self.queries.items(), desc="Processing queries"):
            # Create context-aware prompt
            full_query = f"{query_data['original']}. {query_data['shift']}. Please recommend exactly 3 movies that fit this new context. Do not ask questions."
            
            # Get response from model
            try:
                response = chat_engine.chat(full_query)
                response_text = response.response
            except Exception as e:
                response_text = f"ERROR: {str(e)}"
            
            self.responses[query_id] = {
                'original': query_data['original'],
                'shift': query_data['shift'],
                'context_type': query_data['context_type'],
                'response': response_text
            }
            
            # Reset chat history between queries for independence
            chat_engine.reset()
        
        return self.responses
    
    def display_responses(self):
        
        print("CONTEXTUAL EVALUATION RESPONSES")
        
        
        # Group by context type
        context_types = {
            'temporal': [],
            'mood': [],
            'audience': []
        }
        
        for query_id, data in self.responses.items():
            context_types[data['context_type']].append((query_id, data))
        
        # Display by category
        for context_type, queries in context_types.items():
            if not queries:
                continue
                
            
            print(f"{context_type.upper()} CONTEXT QUERIES")
            
            
            for query_id, data in queries:
                print(f"Query ID: {query_id}")
                print(f"Original: {data['original']}")
                print(f"Context Shift: {data['shift']}")
                print(f"\nModel Response:")
                print(f"{data['response']}")
                
    
    def collect_ratings(self, model_name):

        print(f"RATING RESPONSES FOR: {model_name}")
        
        print("\nFor each response, rate as:")
        print("  1 = SUCCESS (appropriate contextual adaptation)")
        print("  0 = FAILURE (failed to adapt to context)")
        print("  s = SKIP (come back to this later)")
        print("  q = QUIT (exit and save current ratings)")
        print("\n")
        
        ratings = {}
        
        for query_id, data in self.responses.items():
            print(f"\n{'-'*80}")
            print(f"Query ID: {query_id}")
            print(f"Original: {data['original']}")
            print(f"Context Shift: {data['shift']}")
            print(f"\nResponse: {data['response']}")
            print(f"{'-'*80}")
            
            while True:
                rating = input(f"Rate this response [1=Success, 0=Fail, s=Skip, q=Quit]: ").strip().lower()
                
                if rating == 'q': 
                    print("\nExiting rating process...")
                    return ratings  
                elif rating == 's':
                    ratings[query_id] = None
                    break
                elif rating in ['1', '0']:
                    ratings[query_id] = int(rating)
                    break
                else:
                    print("Invalid input. Please enter 1, 0, or s")
        
        return ratings
    
    def save_ratings(self, model_name, ratings):
 
        # Load existing results
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Determine next evaluator ID
        evaluator_id = self._get_next_evaluator_id(all_results)
        
        # Store ratings
        if evaluator_id not in all_results:
            all_results[evaluator_id] = {}
        
        all_results[evaluator_id][model_name] = {
            'timestamp': datetime.now().isoformat(),
            'ratings': ratings,
            'success_count': sum(1 for r in ratings.values() if r == 1),
            'total_rated': sum(1 for r in ratings.values() if r is not None),
            'success_rate': sum(1 for r in ratings.values() if r == 1) / max(sum(1 for r in ratings.values() if r is not None), 1)
        }
        
        # Save to file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nRatings saved as {evaluator_id}")
        print(f"Success rate: {all_results[evaluator_id][model_name]['success_rate']:.2%}")
        print(f"Rated: {all_results[evaluator_id][model_name]['total_rated']}/{len(ratings)} queries")
        
        return evaluator_id
    
    def _get_next_evaluator_id(self, all_results):

        if not all_results:
            return "1.1"
        
        # Get all existing IDs
        existing_ids = list(all_results.keys())
        
        # Parse to find max evaluator and session
        max_evaluator = 0
        sessions_for_max = []
        
        for eval_id in existing_ids:
            parts = eval_id.split('.')
            evaluator_num = int(parts[0])
            session_num = int(parts[1])
            
            if evaluator_num > max_evaluator:
                max_evaluator = evaluator_num
                sessions_for_max = [session_num]
            elif evaluator_num == max_evaluator:
                sessions_for_max.append(session_num)
        
        # Determine next ID
        max_session = max(sessions_for_max) if sessions_for_max else 0
        next_id = f"{max_evaluator}.{max_session + 1}"
        
        return next_id
    
    def get_summary(self):
        
        if not self.results_file.exists():
            print("No evaluation results found.")
            return None
        
        with open(self.results_file, 'r') as f:
            all_results = json.load(f)
        
        # Create summary DataFrame
        summary_data = []
        for evaluator_id, models in all_results.items():
            for model_name, data in models.items():
                summary_data.append({
                    'Evaluator': evaluator_id,
                    'Model': model_name,
                    'Success Rate': f"{data['success_rate']:.2%}",
                    'Successes': data['success_count'],
                    'Total Rated': data['total_rated'],
                    'Timestamp': data['timestamp']
                })
        
        return pd.DataFrame(summary_data)



import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class EvaluationVisualizer:
    """
    Create visualizations for recommendation evaluation results
    Uses Plotly for interactive graphs
    """
    
    def __init__(self, results_file="data/evaluation"):
        """
        Args:
            results_file: Path to directory containing evaluation results
        """
        self.results_file = Path(results_file)
        self.results_data = {}
        
    def load_results(self, model_files):
        """
        Load evaluation results for multiple models
        
        Args:
            model_files: Dict mapping model names to JSON file paths
                        e.g., {'Baseline': 'baseline_metrics.json', ...}
        """
        for model_name, filename in model_files.items():
            filepath = self.results_file / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.results_data[model_name] = data['metrics']
            else:
                print(f"Warning: {filepath} not found")
        
        print(f"Loaded results for {len(self.results_data)} models")
        return self.results_data
    
    def plot_grouped_bars(self, metric_type='HIT', k_values=[1, 3, 5, 10]):
        """
        Create grouped bar chart comparing models for one metric type
        
        Args:
            metric_type: 'HIT', 'MRR', 'NDCG', or 'Recall'
            k_values: List of K values to plot
        """
        fig = go.Figure()
        
        for model_name, metrics in self.results_data.items():
            values = [metrics.get(f'{metric_type}@{k}', 0) for k in k_values]
            fig.add_trace(go.Bar(
                name=model_name,
                x=[f'@{k}' for k in k_values],
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=f'{metric_type} Comparison Across Models',
            xaxis_title='K Value',
            yaxis_title=f'{metric_type} Score',
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_all_metrics_comparison(self, k_values=[1, 3, 5, 10]):
        """
        Create 2x2 subplot with all four metric types
        """
        metrics = ['HIT', 'MRR', 'NDCG', 'Recall']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, metric_type in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            for model_name, model_metrics in self.results_data.items():
                values = [model_metrics.get(f'{metric_type}@{k}', 0) for k in k_values]
                
                fig.add_trace(
                    go.Bar(
                        name=model_name,
                        x=[f'@{k}' for k in k_values],
                        y=values,
                        showlegend=(idx == 0),  # Only show legend once
                        text=[f'{v:.3f}' for v in values],
                        textposition='outside'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Model Performance Across All Metrics",
            height=800,
            showlegend=True,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_metric_vs_k(self, metric_type='HIT', k_values=[1, 3, 5, 10]):
        """
        Line chart showing how metric changes with K
        
        Args:
            metric_type: 'HIT', 'MRR', 'NDCG', or 'Recall'
            k_values: List of K values to plot
        """
        fig = go.Figure()
        
        for model_name, metrics in self.results_data.items():
            values = [metrics.get(f'{metric_type}@{k}', 0) for k in k_values]
            
            fig.add_trace(go.Scatter(
                name=model_name,
                x=k_values,
                y=values,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title=f'{metric_type} Performance vs K',
            xaxis_title='K (Number of Recommendations)',
            yaxis_title=f'{metric_type} Score',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_radar_chart(self, k_value=10):
        """
        Radar chart comparing all models across all metrics at specific K
        
        Args:
            k_value: Which K value to use (default 10)
        """
        metrics = ['HIT', 'MRR', 'NDCG', 'Recall']
        
        fig = go.Figure()
        
        for model_name, model_metrics in self.results_data.items():
            values = [model_metrics.get(f'{metric}@{k_value}', 0) for metric in metrics]
            # Add first value at end to close the radar chart
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max([m.get(f'{metric}@{k_value}', 0) for metric in metrics]) 
                                   for m in self.results_data.values()]) * 1.1]
                )
            ),
            title=f'Model Comparison Across All Metrics @{k_value}',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_heatmap(self, k_values=[1, 3, 5, 10]):
        """
        Heatmap showing all metrics for all models
        """
        # Prepare data
        models = list(self.results_data.keys())
        all_metrics = []
        metric_labels = []
        
        for metric_type in ['HIT', 'MRR', 'NDCG', 'Recall']:
            for k in k_values:
                metric_labels.append(f'{metric_type}@{k}')
                all_metrics.append([
                    self.results_data[model].get(f'{metric_type}@{k}', 0) 
                    for model in models
                ])
        
        # Transpose for correct orientation
        heatmap_data = list(zip(*all_metrics))
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metric_labels,
            y=models,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title='Model Performance Heatmap',
            xaxis_title='Metric',
            yaxis_title='Model',
            height=400,
            width=1000
        )
        
        return fig
    
    def plot_at_k(self, k_value=10):
        """
        Simple bar chart comparing all models at specific K across all metrics
        
        Args:
            k_value: Which K value to use
        """
        metrics = ['HIT', 'MRR', 'NDCG', 'Recall']
        
        fig = go.Figure()
        
        for model_name, model_metrics in self.results_data.items():
            values = [model_metrics.get(f'{metric}@{k_value}', 0) for metric in metrics]
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=metrics,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=f'Model Comparison at K={k_value}',
            xaxis_title='Metric',
            yaxis_title='Score',
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig