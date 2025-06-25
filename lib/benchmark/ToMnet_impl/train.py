import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import platform
import glob

from tomnet import ToMnet, create_tomnet
from data_generation import DataGenerator, ToMnetDataset, collate_fn
from evaluation import evaluate_model, compute_kl_divergence
from typing import Dict, List, Optional, Union


class ExperimentConfig:
    """Configuration for different experiment types"""
    
    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type
        
        if experiment_type == 'figure3':
            self.char_embedding_dim = 2
            self.use_mental_state_net = False
            self.agent_type = 'random'
            self.alpha_values = [0.01, 3.0]
            self.n_agents = 1000
            self.loss_weights = {'action_loss': 1.0}
            self.predictions = ['action']
        elif experiment_type == 'figure5':
            self.char_embedding_dim = 8
            self.use_mental_state_net = True
            self.agent_type = 'goal_directed'
            self.alpha_reward = 0.01
            self.high_cost_ratio = 0.2
            self.n_agents = 40
            self.loss_weights = {
                'action_loss': 1.0,
                'consumption_loss': 0.5,
                'sr_loss': 0.3
            }
            self.predictions = ['action', 'consumption', 'sr']
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")


class ToMnetTrainer:
    """Unified trainer for ToMnet experiments"""
    
    def __init__(self, model: ToMnet, config: ExperimentConfig, device: str = 'cuda', learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Loss weights from config
        self.loss_weights = config.loss_weights
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {}
        n_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                batch['past_trajectories'],
                batch['current_trajectory'],
                batch['current_state']
            )
            
            # Compute losses
            targets = {}
            if 'true_actions' in batch:
                targets['true_actions'] = batch['true_actions']
            if 'true_consumption' in batch:
                targets['true_consumption'] = batch['true_consumption']
            if 'true_sr' in batch:
                targets['true_sr'] = batch['true_sr']
            
            losses = self.model.compute_loss(predictions, targets)
            
            # Weighted total loss
            weighted_loss = 0
            for loss_name, loss_value in losses.items():
                if loss_name != 'total_loss':
                    weight = self.loss_weights.get(loss_name, 1.0)
                    weighted_loss += weight * loss_value
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name not in total_losses:
                    total_losses[loss_name] = 0
                total_losses[loss_name] += loss_value.item()
            
            total_losses['weighted_loss'] = total_losses.get('weighted_loss', 0) + weighted_loss.item()
            n_batches += 1
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_losses = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(
                    batch['past_trajectories'],
                    batch['current_trajectory'],
                    batch['current_state']
                )
                
                # Compute losses
                targets = {}
                if 'true_actions' in batch:
                    targets['true_actions'] = batch['true_actions']
                if 'true_consumption' in batch:
                    targets['true_consumption'] = batch['true_consumption']
                if 'true_sr' in batch:
                    targets['true_sr'] = batch['true_sr']
                
                losses = self.model.compute_loss(predictions, targets)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in total_losses:
                        total_losses[loss_name] = 0
                    total_losses[loss_name] += loss_value.item()
                
                n_batches += 1
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'loss_weights': self.loss_weights
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']


def create_model(experiment_type: str, state_dim: int, config: ExperimentConfig) -> ToMnet:
    """Create ToMnet model based on experiment type"""
    return create_tomnet(
        experiment_type=experiment_type,
        state_dim=state_dim,
        char_embedding_dim=config.char_embedding_dim
    )


def generate_data(experiment_type: str, config: ExperimentConfig, args) -> Union[Dict, Dict[str, Dict]]:
    """Generate training data based on experiment type"""
    # Check if data files already exist
    data_files = glob.glob("data/figure*.pkl")
    if data_files:
        print(f"Found existing data files: {data_files}")
        print("Skipping data generation...")
        
        # Load existing data files
        if experiment_type == 'figure3':
            datasets = {}
            alpha_values = getattr(args, 'alpha_values', config.alpha_values)
            
            for alpha in alpha_values:
                file_path = f"data/figure3_alpha_{alpha}.pkl"
                if os.path.exists(file_path):
                    print(f"Loading existing data for alpha={alpha}")
                    import pickle
                    with open(file_path, 'rb') as f:
                        datasets[alpha] = pickle.load(f)
                else:
                    print(f"Warning: Expected file {file_path} not found, generating new data")
                    # Generate missing data
                    data_generator = DataGenerator()
                    dataset = data_generator.generate_random_agent_data(
                        n_agents=args.n_agents or config.n_agents,
                        n_episodes_per_agent=args.n_episodes_per_agent,
                        alpha=alpha,
                        save_path=file_path,
                        n_workers=args.n_workers
                    )
                    datasets[alpha] = dataset
            
            # Create mixed dataset if requested
            if getattr(args, 'mixed_training', False):
                print("Creating mixed dataset")
                mixed_data = []
                for alpha, dataset in datasets.items():
                    mixed_data.extend(dataset['data'])
                
                mixed_dataset = {
                    'data': mixed_data,
                    'meta': {
                        'mixed': True,
                        'alpha_values': list(datasets.keys()),
                        'state_dim': datasets[list(datasets.keys())[0]]['meta']['state_dim']
                    }
                }
                datasets['mixed'] = mixed_dataset
            
            return datasets
            
        elif experiment_type == 'figure5':
            file_path = "data/figure5_data.pkl"
            if os.path.exists(file_path):
                print("Loading existing figure5 data")
                import pickle
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print("Figure5 data file not found, generating new data")
                # Fall through to generate new data
    
    # Generate new data if not found
    data_generator = DataGenerator()
    
    if experiment_type == 'figure3':
        # Generate datasets for different alpha values
        datasets = {}
        alpha_values = getattr(args, 'alpha_values', config.alpha_values)
        
        for alpha in alpha_values:
            print(f"Generating data for alpha={alpha}")
            dataset = data_generator.generate_random_agent_data(
                n_agents=args.n_agents or config.n_agents,
                n_episodes_per_agent=args.n_episodes_per_agent,
                alpha=alpha,
                save_path=f"data/figure3_alpha_{alpha}.pkl",
                n_workers=args.n_workers
            )
            datasets[alpha] = dataset
        
        # Create mixed dataset if requested
        if getattr(args, 'mixed_training', False):
            print("Creating mixed dataset")
            mixed_data = []
            for alpha, dataset in datasets.items():
                mixed_data.extend(dataset['data'])
            
            mixed_dataset = {
                'data': mixed_data,
                'meta': {
                    'mixed': True,
                    'alpha_values': list(datasets.keys()),
                    'state_dim': datasets[list(datasets.keys())[0]]['meta']['state_dim']
                }
            }
            datasets['mixed'] = mixed_dataset
        
        return datasets
    
    elif experiment_type == 'figure5':
        # Generate goal-directed agent data
        dataset = data_generator.generate_goal_directed_agent_data(
            n_agents=args.n_agents or config.n_agents,
            n_episodes_per_agent=args.n_episodes_per_agent,
            alpha_reward=config.alpha_reward,
            high_cost_ratio=config.high_cost_ratio,
            save_path="data/figure5_data.pkl",
            n_workers=args.n_workers
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def train_unified_model(experiment_type: str, args):
    """Unified training function for both Figure 3 and Figure 5 experiments"""
    print(f"Training ToMnet for {experiment_type.title()} Experiment")
    
    # Create experiment configuration
    config = ExperimentConfig(experiment_type)
    
    # Generate data
    data = generate_data(experiment_type, config, args)
    
    if experiment_type == 'figure3':
        # Train separate models for each dataset
        results = {}
        datasets = data  # data is a dict of datasets for figure3
        
        for dataset_name, dataset in datasets.items():
            print(f"\nTraining model for {dataset_name}")
            
            # Create model
            state_dim = dataset['meta']['state_dim']
            model = create_model(experiment_type, state_dim, config)
            
            # Create dataset and dataloader
            train_dataset = ToMnetDataset(dataset, experiment_type=experiment_type)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # Create trainer
            trainer = ToMnetTrainer(model, config, args.device, args.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            
            for epoch in range(args.n_epochs):
                # Train
                train_losses = trainer.train_epoch(train_loader)
                
                # Log progress
                print(f"Epoch {epoch+1}/{args.n_epochs}")
                print(f"Train Loss: {train_losses['total_loss']:.4f}")
                
                # Save best model
                if train_losses['total_loss'] < best_val_loss:
                    best_val_loss = train_losses['total_loss']
                    save_path = f"models/{experiment_type}_{dataset_name}_best.pth"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    trainer.save_checkpoint(epoch, best_val_loss, save_path)
            
            results[dataset_name] = {
                'best_val_loss': best_val_loss,
                'model_path': save_path
            }
        
        return results
    
    elif experiment_type == 'figure5':
        # Single model training with train/val split
        dataset = data  # data is a single dataset for figure5
        
        # Split into train/val
        data_list = dataset['data']
        split_idx = int(0.8 * len(data_list))
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
        
        train_dataset_dict = {'data': train_data, 'meta': dataset['meta']}
        val_dataset_dict = {'data': val_data, 'meta': dataset['meta']}
        
        # Create model
        state_dim = dataset['meta']['state_dim']
        model = create_model(experiment_type, state_dim, config)
        
        # Create datasets and dataloaders
        train_dataset = ToMnetDataset(train_dataset_dict, experiment_type=experiment_type)
        val_dataset = ToMnetDataset(val_dataset_dict, experiment_type=experiment_type)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Create trainer
        trainer = ToMnetTrainer(model, config, args.device, args.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(args.n_epochs):
            # Train
            train_losses = trainer.train_epoch(train_loader)
            
            # Validate
            val_losses = trainer.validate(val_loader)
            
            # Scheduler step
            trainer.scheduler.step(val_losses['total_loss'])
            
            # Log progress
            print(f"Epoch {epoch+1}/{args.n_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            # Save best model
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                save_path = f"models/{experiment_type}_best.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                trainer.save_checkpoint(epoch, best_val_loss, save_path)
        
        return {
            'best_val_loss': best_val_loss,
            'model_path': save_path
        }


def main():
    parser = argparse.ArgumentParser(description='Train ToMnet')
    parser.add_argument('--experiment', choices=['figure3', 'figure5', 'both'], 
                       default='both', help='Which experiment to run')
    # Detect device based on platform
    if platform.system() == "Darwin":  # macOS
        default_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser.add_argument('--device', default=default_device, help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--n_episodes_per_agent', type=int, default=100, help='Episodes per agent')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of parallel workers for data generation')
    
    # Figure 3 specific
    parser.add_argument('--alpha_values', nargs='+', type=float, default=[0.01, 3.0],
                       help='Alpha values for Figure 3')
    parser.add_argument('--mixed_training', action='store_true',
                       help='Train on mixed alpha dataset')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results = {}
    
    if args.experiment in ['figure3', 'both']:
        results['figure3'] = train_unified_model('figure3', args)
    
    if args.experiment in ['figure5', 'both']:
        results['figure5'] = train_unified_model('figure5', args)
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed!")
    print(f"Results saved to training_results.json")


if __name__ == '__main__':
    main()