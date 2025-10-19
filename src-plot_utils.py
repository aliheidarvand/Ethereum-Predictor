import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_scientific_charts(train_loss, test_rmse, actuals, preds):
    residuals = preds - actuals
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch'); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(epochs, test_rmse, marker='o', label='Test RMSE', color='tab:orange')
    plt.xlabel('Epoch'); plt.ylabel('RMSE')
    plt.title('Test RMSE vs. Epoch'); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(actuals, label='Actual', linewidth=1.5)
    plt.plot(preds, label='Predicted', linewidth=1.5)
    plt.xlabel('Time Index'); plt.ylabel('Price (USD)')
    plt.title('Actual vs. Predicted Price'); plt.legend(); plt.grid(True)

    plt.figure()
    plt.scatter(actuals, preds, alpha=0.5)
    minv, maxv = min(np.min(actuals), np.min(preds)), max(np.max(actuals), np.max(preds))
    plt.plot([minv, maxv], [minv, maxv], linestyle='--', linewidth=1)
    corr, _ = pearsonr(actuals.flatten(), preds.flatten())
    plt.xlabel('Actual Price (USD)'); plt.ylabel('Predicted Price (USD)')
    plt.title(f'Predicted vs. Actual (Pearson r = {corr:.2f})'); plt.grid(True)

    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residual (Predicted - Actual)'); plt.ylabel('Frequency')
    plt.title('Distribution of Residuals'); plt.grid(True)

    plt.figure()
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted Price (USD)'); plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Residuals vs. Predicted Values'); plt.grid(True)
