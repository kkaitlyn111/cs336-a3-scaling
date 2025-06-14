import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

data_file = "data/isoflops_curves.json"

def load_data(data_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    best_loss_per_budget = {}
    for run in data:
        budget = run["compute_budget"]
        loss = run["final_loss"]
        N = run["parameters"]

        if budget not in best_loss_per_budget:
            best_loss_per_budget[budget] = (N, loss)
        else:
            if loss < best_loss_per_budget[budget][1]:
                best_loss_per_budget[budget] = (N, loss)

    budgets = np.array(sorted(best_loss_per_budget.keys()))
    N_opt = np.array([best_loss_per_budget[budget][0] for budget in budgets])
    D_opt = np.array([budget / (6 * best_loss_per_budget[budget][0]) for budget in budgets])

    return budgets, N_opt, D_opt, best_loss_per_budget

# Power law relationship: y = 10^a * x^b
def power_law_model(x, a, b):
    return a + b * x

def fit_power_law(x_data, y_data, initial_guess=None):
    if initial_guess is None:
        initial_guess = [1.0, 0.5]
    
    # Convert to log space for linear fitting
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)
    
    # Fit the linear relationship in log space
    fitted_params, covariance = curve_fit(power_law_model, log_x, log_y, p0=initial_guess)
    a_fit, b_fit = fitted_params
    
    # Calculate uncertainties
    uncertainties = np.sqrt(np.diag(covariance))
    
    print(f"Fitted parameters: a = {a_fit:.4f} ± {uncertainties[0]:.4f}")
    print(f"Scaling exponent: b = {b_fit:.4f} ± {uncertainties[1]:.4f}")
    
    # Make predictions for larger compute budgets
    test_budgets = [1e23, 1e24]
    predictions = {}
    for budget in test_budgets:
        predicted_log = power_law_model(np.log10(budget), a_fit, b_fit)
        predicted_value = 10**predicted_log
        predictions[budget] = predicted_value
        print(f"Prediction at C={budget:.0e}: {predicted_value:,.0f}")
    
    return fitted_params, covariance, predictions

def create_scaling_plot(x_data, y_data, fitted_params, x_label, y_label, title, filename):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    
    # Plot original data points
    ax.loglog(x_data, y_data, 'o', markersize=8, label='Experimental data points', alpha=0.8, color='blue')
    
    # Create smooth curve for the fitted model
    x_smooth = np.logspace(np.log10(x_data.min()), np.log10(1e24), 200)
    log_x_smooth = np.log10(x_smooth)
    log_y_smooth = power_law_model(log_x_smooth, *fitted_params)
    y_smooth = 10**log_y_smooth
    
    ax.loglog(x_smooth, y_smooth, '-', linewidth=3, label=f'Scaling law fit: y ∝ x^{fitted_params[1]:.3f}', color='green')
    
    # Highlight extrapolation points
    extrapolation_budgets = [1e23, 1e24]
    extrapolation_values = [10**power_law_model(np.log10(b), *fitted_params) for b in extrapolation_budgets]
    ax.scatter(extrapolation_budgets, extrapolation_values, 
               color='red', s=150, zorder=5, label='Extrapolated predictions', marker='*')
    
    # Add text annotations for extrapolation points
    for i, (budget, value) in enumerate(zip(extrapolation_budgets, extrapolation_values)):
        ax.annotate(f'C={budget:.0e}\n{value:,.0f}', 
                   xy=(budget, value), xytext=(10, 10), 
                   textcoords='offset points', ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=10)
    
    # Customize the plot
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to show full range
    ax.set_xlim(1e18, 1e25)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as {filename}")

def main():
    # Load and process the data
    budgets, N_opt, D_opt, raw_data = load_data(data_file)
    
    print("=== Parameter Scaling Analysis ===")
    param_params, param_cov, param_predictions = fit_power_law(budgets, N_opt)
    create_scaling_plot(budgets, N_opt, param_params, 
                       "Compute Budget C (FLOPs)", "Optimal Parameters N_opt", 
                       "Model Size Scaling Law", "parameter_scaling.png")
    
    print("\n=== Data Scaling Analysis ===")
    data_params, data_cov, data_predictions = fit_power_law(budgets, D_opt)
    create_scaling_plot(budgets, D_opt, data_params, 
                       "Compute Budget C (FLOPs)", "Optimal Data D_opt", 
                       "Dataset Size Scaling Law", "data_scaling.png")
    
    # Verify Chinchilla ratio
    print("\n=== Chinchilla Ratio Verification ===")
    for i, budget in enumerate(budgets):
        ratio = budget / (6 * N_opt[i])
        print(f"C={budget:.0e}: C/(6N) = {ratio:.2e}, D_opt = {D_opt[i]:.2e}")
    
    # Deliverable responses
    print("\n" + "="*60)
    print("DELIVERABLES:")
    print("="*60)
    
    print(f"\n1. Model Size Scaling Law:")
    print(f"   Predicted optimal model size for 10^23 FLOPs: {param_predictions[1e23]:,.0f} parameters")
    print(f"   Predicted optimal model size for 10^24 FLOPs: {param_predictions[1e24]:,.0f} parameters")
    print(f"   One-sentence response: The optimal model size scales as N_opt ∝ C^{param_params[1]:.3f}, predicting {param_predictions[1e23]:,.0f} parameters for 10^23 FLOPs and {param_predictions[1e24]:,.0f} parameters for 10^24 FLOPs.")
    
    print(f"\n2. Dataset Size Scaling Law:")
    print(f"   Predicted optimal dataset size for 10^23 FLOPs: {data_predictions[1e23]:,.0f} tokens")
    print(f"   Predicted optimal dataset size for 10^24 FLOPs: {data_predictions[1e24]:,.0f} tokens")
    print(f"   One-sentence response: The optimal dataset size scales as D_opt ∝ C^{data_params[1]:.3f}, predicting {data_predictions[1e23]:,.0f} tokens for 10^23 FLOPs and {data_predictions[1e24]:,.0f} tokens for 10^24 FLOPs.")

if __name__ == "__main__":
    main()