import matplotlib.pyplot as plt
import statsmodels.api as sm

def residuals_plot(target, residuals):
    plt.scatter(target, residuals)
    plt.xlabel('Target')
    plt.ylabel('Residual')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual vs. Target')
    plt.savefig('export/residuals_plot.png')

def qq_residuals_plot(residuals):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    sm.qqplot(residuals, line='q', ax=ax, fit=True)
    plt.title('Q-Q plot of the residuals')
    plt.savefig('export/qq_residuals_plot.png')
    plt.close(fig)