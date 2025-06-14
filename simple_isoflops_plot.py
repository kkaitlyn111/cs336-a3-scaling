#!/usr/bin/env python3
"""
Simple Isoflops Curves Visualization

This script creates key visualizations for the isoflops curves data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(file_path='data/isoflops_curves.json'):
    """Load the isoflops curves data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_isoflops_curves(data):
    """Create the main isoflops curves plot."""
    plt.figure(figsize=(12, 8))
    
    # Group data by compute budget
    compute_budgets = sorted(list(set(item['compute_budget'] for item in data)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(compute_budgets)))
    
    for i, budget in enumerate(compute_budgets):
        budget_data = [item for item in data if item['compute_budget'] == budget]
        # Sort by parameters for proper line plotting
        budget_data.sort(key=lambda x: x['parameters'])
        
        parameters = [item['parameters'] for item in budget_data]
        losses = [item['final_loss'] for item in budget_data]
        
        # Plot line with markers
        plt.plot(parameters, losses, 
                color=colors[i], 
                marker='o', 
                markersize=6,
                linewidth=2,
                label=f'Compute Budget: {budget:.0e}')
    
    plt.xscale('log')
    plt.xlabel('N')
    plt.ylabel('Final Loss')
    plt.title('Isoflops Curves: Loss vs N by C')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('isoflops_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimal_scaling(data):
    """Plot optimal model size and loss vs compute budget."""
    compute_budgets = sorted(list(set(item['compute_budget'] for item in data)))
    optimal_params = []
    optimal_losses = []
    
    for budget in compute_budgets:
        budget_data = [item for item in data if item['compute_budget'] == budget]
        # Find the point with minimum loss for this compute budget
        min_loss_point = min(budget_data, key=lambda x: x['final_loss'])
        optimal_params.append(min_loss_point['parameters'])
        optimal_losses.append(min_loss_point['final_loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot optimal parameters
    ax1.plot(compute_budgets, optimal_params, 'bo-', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Compute Budget')
    ax1.set_ylabel('Optimal Number of Parameters')
    ax1.set_title('Optimal Model Size vs Compute Budget')
    ax1.grid(True, alpha=0.3)
    
    # Plot optimal losses
    ax2.plot(compute_budgets, optimal_losses, 'ro-', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Compute Budget')
    ax2.set_ylabel('Best Achievable Loss')
    ax2.set_title('Best Achievable Loss vs Compute Budget')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary(data):
    """Print a summary of the data."""
    print("=== Isoflops Curves Data Summary ===")
    print(f"Total data points: {len(data)}")
    
    parameters = [item['parameters'] for item in data]
    compute_budgets = [item['compute_budget'] for item in data]
    losses = [item['final_loss'] for item in data]
    
    print(f"\nParameters range: {min(parameters):.0e} - {max(parameters):.0e}")
    print(f"Compute budgets: {len(set(compute_budgets))} unique values")
    print(f"Loss range: {min(losses):.3f} - {max(losses):.3f}")
    
    print(f"\nCompute budget distribution:")
    for budget in sorted(set(compute_budgets)):
        count = len([item for item in data if item['compute_budget'] == budget])
        print(f"  {budget:.0e}: {count} data points")

def main():
    """Main function."""
    print("Loading isoflops curves data...")
    data = load_data()
    
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    
    # Create plots
    plot_isoflops_curves(data)
    plot_optimal_scaling(data)
    
    # Print summary
    print_data_summary(data)
    
    print("\nVisualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    main() 