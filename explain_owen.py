import argparse
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from models import *
from datasets import *
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import reload_shap_plot

def parse_args():
    parser = argparse.ArgumentParser(description='Explain model predictions using SHAP')
    parser.add_argument('--dataset', type=str, default='Node123_loop', 
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=['cnn', 'cnn_pairwise', 'mlp', 'transformer', 'lstm'], 
                        help='Model type')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory to load model weights from')
    parser.add_argument('--train_size', type=float, default=0.8, 
                        help='Fraction of data used for training')
    parser.add_argument('--seed', type=int, default=20, 
                        help='Random seed for dataset splitting')
    parser.add_argument('--output_dir', type=str, default='shap_plots', 
                        help='Directory to save output plots')
    return parser.parse_args()

def load_dataset_and_model(args):
    # Load dataset
    dataset = BinaryDataset(args.dataset)
    n_buses = dataset.n_buses
    
    # Split dataset
    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Feature names
    feature_names = [f'Bus {i+1}' for i in range(1, n_buses + 1)]
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'cnn':
        model = CnnClassifier().to(device)
    elif args.model == 'cnn_pairwise':
        model = CnnClassifierPairwise().to(device)
    elif args.model == 'mlp':
        model = MLPClassifier(dataset.n_buses).to(device)
    elif args.model == 'transformer':
        model = TransformerClassifier(dataset.n_buses).to(device)
    elif args.model == 'lstm':
        model = LSTMClassifier(dataset.n_buses).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(f'{args.checkpoint_dir}/{args.dataset}/{args.model}_model.pth', weights_only=True))
    model.eval()
    
    return dataset, train_dataset, test_dataset, feature_names, model, device

def prepare_data(train_dataset):
    label_1_samples = []
    label_0_samples = []
    for data, label in train_dataset:
        if label.item() == 1:  # Filter label 1 samples
            label_1_samples.append(data)
        else: 
            label_0_samples.append(data)
    
    # Convert to NumPy arrays
    label_0_samples = np.array(label_0_samples)
    label_1_samples = np.array(label_1_samples)
    
    print(f"Label 0 samples shape: {label_0_samples.shape}")
    print(f"Label 1 samples shape: {label_1_samples.shape}")
    
    return label_0_samples, label_1_samples

def prepare_background(label_0_samples):
    # Compute mean background from normal (label 0) samples
    background_mean = label_0_samples.mean(0)
    background = background_mean
    background = np.tile(background, (2, 1))  # Expand background size *2
    
    print(f"Background shape: {background.shape}")
    return background

def get_matrix_from_data(data_normal, data_outage):
    combined_data = data_normal
    corr_matrix = combined_data.corr()
    distance_matrix = 1 - corr_matrix
    Z = linkage(squareform(distance_matrix), method='average')
    return Z, corr_matrix

def plot_dendrogram(Z, corr_matrix, output_dir):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=corr_matrix.columns, leaf_rotation=45)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Feature')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
    plt.close()

def create_model_wrapper(model, device):
    def model_wrapper(x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            probability = torch.sigmoid(model(x))  # Convert logits to probabilities
            prediction = (probability > 0.5).float()  # Convert to binary labels
        return prediction.cpu().numpy().squeeze()
    return model_wrapper

def create_shap_explainer(wrapper_fn, background, Z, feature_names):
    masker = shap.maskers.Partition(background)
    masker.clustering = Z  # Set partition tree
    explainer = shap.PartitionExplainer(wrapper_fn, masker, feature_names=feature_names)
    return explainer

def generate_shap_plots(explainer, label_1_samples, output_dir):
    # Compute SHAP values
    shap_values = explainer(label_1_samples.squeeze())
    
    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=6, plot_size=(7.4, 2), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'))
    plt.close()
    
    # Waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values.mean(0), show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'))
    plt.close()
    
    # Heatmap
    plt.figure(figsize=(10, 6))
    shap.plots.heatmap(shap_values, max_display=6, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_heatmap.png'))
    plt.close()
    
    return shap_values

def plot_clustering_tree(explainer, output_dir):
    clustering_tree = explainer._clustering
    
    if clustering_tree is not None:
        plt.figure(figsize=(12, 6))
        dendrogram(clustering_tree, labels=explainer.feature_names, leaf_rotation=45)
        plt.xlabel("Feature Name")
        plt.ylabel("Clustering Distance")
        plt.title("PartitionExplainer Auto-Generated Feature Clustering Tree")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_tree.png'))
        plt.close()
    else:
        print("PartitionExplainer did not generate a clustering tree.")

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    output_dir = f'{args.output_dir}/{args.model}_{args.dataset}'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load dataset and model
    dataset, train_dataset, test_dataset, feature_names, model, device = load_dataset_and_model(args)
    print(f"Loaded {args.dataset} dataset with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print(f"Using model from {args.checkpoint_dir}/{args.model}_model.pth")
    
    # Filter and prepare data
    label_0_samples, label_1_samples = prepare_data(train_dataset)
    
    # Prepare background for SHAP
    background = prepare_background(label_0_samples)
    
    # Generate correlation matrix and hierarchical clustering
    Z, corr_matrix = get_matrix_from_data(dataset.data_df, dataset.data_outage_df)
    plot_dendrogram(Z, corr_matrix, output_dir)
    
    # Create model wrapper for SHAP
    wrapper_fn = create_model_wrapper(model, device)
    
    # Test model wrapper
    instance = label_1_samples[0]
    prediction = wrapper_fn(instance)
    print(f"Model prediction for sample instance: {prediction}")
    
    # Create SHAP explainer
    explainer = create_shap_explainer(wrapper_fn, background, Z, feature_names)
    plot_clustering_tree(explainer, output_dir)
    
    # Generate SHAP plots
    shap_values = generate_shap_plots(explainer, label_1_samples, output_dir)
    print(f"SHAP analysis completed. Plots saved to {output_dir}")
