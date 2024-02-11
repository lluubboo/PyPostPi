import matplotlib.pyplot as plt

def residuals_plot(target, residuals):
    plt.scatter(target, residuals)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual vs. Predicted')
    plt.savefig('export/residuals_plot.png')