from scipy import stats as scipy_stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv('single_stage_cv_SA_results.csv')
y = df['built_up_area']
x = df['built_up_area_pred']

mask = ~(x.isna() | y.isna() | ~np.isfinite(x) | ~np.isfinite(y))
x, y = x[mask], y[mask]

zero_mask = (x == 0) | (y == 0)
zero_x, zero_y = x[zero_mask], y[zero_mask]
nonzero_x, nonzero_y = x[~zero_mask], y[~zero_mask]


x_np = np.array(x)
y_np = np.array(y)
slope, intercept, r_value, p_value, std_err_reg = scipy_stats.linregress(x_np, y_np)

R2 = r2_score(y_np, x_np)
MAE = mean_absolute_error(y_np, x_np)
RMSE = np.sqrt(mean_squared_error(y_np, x_np))
n = len(x_np)
p = 37
Adjusted_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)

print(f"R²: {R2:.4f}")
print(f"Adjusted R²: {Adjusted_R2:.4f}")
print(f"MAE: {MAE:.4f}")
print(f"RMSE: {RMSE:.4f}")


max_plot_points = 50000
if len(nonzero_x) > max_plot_points:
    sample_indices = np.random.choice(len(nonzero_x), size=max_plot_points, replace=False)
    nonzero_x_sample = nonzero_x.iloc[sample_indices]
    nonzero_y_sample = nonzero_y.iloc[sample_indices]
else:
    nonzero_x_sample = nonzero_x
    nonzero_y_sample = nonzero_y

if len(nonzero_x_sample) > 1 and nonzero_x_sample.var() > 0 and nonzero_y_sample.var() > 0:
    scaler = StandardScaler()
    nonzero_data = np.column_stack([nonzero_x_sample, nonzero_y_sample])
    nonzero_data_standardized = scaler.fit_transform(nonzero_data)
    xy_nonzero_standardized = nonzero_data_standardized.T
    z_nonzero = scipy_stats.gaussian_kde(xy_nonzero_standardized)(xy_nonzero_standardized)
    idx_nonzero = z_nonzero.argsort()
    nonzero_x_sorted = nonzero_x_sample.iloc[idx_nonzero]
    nonzero_y_sorted = nonzero_y_sample.iloc[idx_nonzero]
    z_nonzero_sorted = z_nonzero[idx_nonzero]
else:
    nonzero_x_sorted = nonzero_x_sample
    nonzero_y_sorted = nonzero_y_sample
    z_nonzero_sorted = np.ones(len(nonzero_x_sample))

max_val = max(x_np.max(), y_np.max())
scale = max_val * 1.1
margin = scale * 0.02

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

if len(nonzero_x_sorted) > 0:
   scatter = ax.scatter(nonzero_x_sorted, nonzero_y_sorted, marker='o',
                       c=z_nonzero_sorted, edgecolors=None, s=15, 
                       cmap='RdBu_r', alpha=0.8)
   cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', 
                      extend='both', pad=0.015, aspect=30, 
                      label='Density')

ax.grid(True, linestyle='--', alpha=0.2)

text_x = scale * 0.95
text_y_start = scale * 0.2
text_spacing = scale * 0.05
plt.text(text_x, text_y_start, f'$R^2={R2:.2f}$', 
        family='Arial', horizontalalignment='right')
plt.text(text_x, text_y_start - text_spacing,
         f'$Adjusted\\;R^2 = {Adjusted_R2:.2f}$',
         family='Arial', horizontalalignment='right')
plt.text(text_x, text_y_start - 2*text_spacing, f'$MAE={MAE:.2f}$', 
        family='Arial', horizontalalignment='right')
plt.text(text_x, text_y_start - 3*text_spacing, f'$RMSE={RMSE:.2f}$', 
        family='Arial', horizontalalignment='right')


plt.xlim(-margin, scale)
plt.ylim(-margin, scale)

ax.set_xticks(np.concatenate([[0], ax.get_xticks()[ax.get_xticks() > 0]]))
ax.set_yticks(np.concatenate([[0], ax.get_yticks()[ax.get_yticks() > 0]]))

final_xlim = ax.get_xlim()
final_ylim = ax.get_ylim()
final_max = min(final_xlim[1], final_ylim[1])

plt.plot([0, final_max], [0, final_max], 'red', lw=1.5, linestyle='--', label='1:1 line')
x_range = np.array([0, final_xlim[1]])
regression_line_full = slope * x_range + intercept
plt.plot(x_range, regression_line_full, 'black', lw=1.5, label='Regression Line')

ax.legend(loc='upper left', frameon=False)
plt.xlabel(f'{x.name}')
plt.ylabel(f'{y.name}')
plt.title('Density Plot')
plt.tight_layout()
plt.savefig('density_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()
