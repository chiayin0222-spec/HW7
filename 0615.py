import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load Boston Housing Dataset
print("Loading Boston Housing Dataset...")
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
df = pd.concat([raw_df.iloc[::2, :].reset_index(drop=True), 
                raw_df.iloc[1::2, :3].reset_index(drop=True)], axis=1)
columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df.columns = columns

# 2. Initial Exploration
print("\n--- Initial Exploration ---")
print(df.head())
print(df.info())
print(df.describe())

# 3. Check Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Feature / Target Split
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# 5. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Standardization
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# 7. Model Training & Evaluation (10 Algorithms)
print("\nTraining 10 Regression Algorithms...")

# Initialize models
models_10 = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=10.0, epsilon=0.1),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
}

# 8. Evaluation Function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Train and evaluate all 10
y_preds = {}
results_10 = {}

print("\n--- Model Evaluation Results ---")
for name, model in models_10.items():
    model.fit(X_train_scaled, y_train)
    y_preds[name] = model.predict(X_test_scaled)
    mae, rmse, r2 = evaluate(y_test, y_preds[name])
    results_10[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n{name}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")

# Keep references for downstream tasks
rf = models_10["Random Forest"]
xgb = models_10["XGBoost"]
y_pred_lr = y_preds["Linear Regression"]
y_pred_rf = y_preds["Random Forest"]
y_pred_xgb = y_preds["XGBoost"]

# 12. Feature Importance
print("\n--- Feature Importance ---")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance.head(10))

# 13. Visualizations (Save to files since non-interactive)
print("\nSaving plots...")

# Plot 1: Feature Importance
plt.figure(figsize=(8,5))
plt.bar(importance["Feature"].head(10), importance["Importance"].head(10))
plt.title("Top 10 Important Features")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Plot 3: RM vs MEDV
plt.figure(figsize=(8,5))
plt.scatter(df["RM"], df["MEDV"], alpha=0.7)
plt.xlabel("Average Rooms (RM)")
plt.ylabel("House Price (MEDV)")
plt.title("RM vs MEDV")
plt.tight_layout()
plt.savefig("rm_vs_medv.png")
plt.close()

# Plot 4: LSTAT vs MEDV
plt.figure(figsize=(8,5))
plt.scatter(df["LSTAT"], df["MEDV"], alpha=0.7, color='red')
plt.xlabel("LSTAT")
plt.ylabel("House Price (MEDV)")
plt.title("LSTAT vs MEDV")
plt.tight_layout()
plt.savefig("lstat_vs_medv.png")
# Plot 5: Top 10 Features Linear Regression Curves
top10_features = ["RM", "LSTAT", "DIS", "CRIM", "PTRATIO", "NOX", "TAX", "AGE", "B", "INDUS"]
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for i, col in enumerate(top10_features):
    sns.regplot(
        x=df[col], 
        y=df["MEDV"], 
        ax=axes[i], 
        scatter_kws={"alpha": 0.5, "color": "blue"}, 
        line_kws={"color": "red"}
    )
    axes[i].set_title(f"Linear Fit: {col} vs MEDV")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("House Price (MEDV)")

plt.tight_layout()
plt.savefig("top10_linear_curves.png")
plt.close()

# Plot 7: 10 Models R^2 Comparison
names = list(results_10.keys())
r2_scores = [results_10[name]["R2"] for name in names]

# Sort by R2 descending
sorted_indices = np.argsort(r2_scores)[::-1]
sorted_names = [names[i] for i in sorted_indices]
sorted_r2 = [r2_scores[i] for i in sorted_indices]

plt.figure(figsize=(12, 6))
colors = sns.color_palette("viridis", len(sorted_names))
bars = plt.barh(sorted_names, sorted_r2, color=colors)
plt.xlabel("R^2 Score (Higher is Better)", fontsize=11)
plt.title("R^2 Score Comparison of 10 Regression Algorithms", fontsize=14, fontweight='bold', pad=15)
plt.xlim(0, 1.0)
plt.gca().invert_yaxis()  # Best model at the top

# Add values to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             va='center', ha='left', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("models_r2_comparison.png", dpi=200)
plt.close()



# Plot 6: Feature Selection Schemes Comparison (RMSE & R-squared + Table)
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

# Selection helper functions
def get_stepwise_features(X_tr, y_tr, k):
    selected = []
    remaining = list(X_tr.columns)
    for _ in range(k):
        best_score = float('inf')
        best_feat = None
        for feat in remaining:
            candidate = selected + [feat]
            model = LinearRegression()
            scores = cross_val_score(model, X_tr[candidate], y_tr, cv=5, scoring='neg_mean_squared_error')
            score = -scores.mean()
            if score < best_score:
                best_score = score
                best_feat = feat
        if best_feat:
            selected.append(best_feat)
            remaining.remove(best_feat)
    return selected

def get_pearson_features(df_tr, k):
    corrs = df_tr.corr()["MEDV"].abs().drop("MEDV")
    selected = corrs.sort_values(ascending=False).index[:k].tolist()
    return selected

def get_rfe_features(X_tr, y_tr, k):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit(X_tr, y_tr)
    selected = [X_tr.columns[i] for i in range(len(X_tr.columns)) if selector.support_[i]]
    return selected

def get_lasso_features(X_tr, y_tr, k):
    lasso = LassoCV(cv=5, random_state=42).fit(X_tr, y_tr)
    coefs = pd.Series(np.abs(lasso.coef_), index=X_tr.columns)
    selected = coefs.sort_values(ascending=False).index[:k].tolist()
    return selected

def get_rf_features(X_tr, y_tr, k):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    importances = pd.Series(rf.feature_importances_, index=X_tr.columns)
    selected = importances.sort_values(ascending=False).index[:k].tolist()
    return selected

methods = ["Stepwise Manual", "Pearson Correlation", "RFE (Recursive)", "Lasso L1", "Random Forest"]
comp_results = {m: {"rmse": [], "r2": [], "features": []} for m in methods}
train_df = pd.concat([X_train, y_train], axis=1)

for k in range(1, 11):
    comp_results["Stepwise Manual"]["features"].append(get_stepwise_features(X_train_scaled, y_train, k))
    comp_results["Pearson Correlation"]["features"].append(get_pearson_features(train_df, k))
    comp_results["RFE (Recursive)"]["features"].append(get_rfe_features(X_train_scaled, y_train, k))
    comp_results["Lasso L1"]["features"].append(get_lasso_features(X_train_scaled, y_train, k))
    comp_results["Random Forest"]["features"].append(get_rf_features(X_train, y_train, k))

for m in methods:
    for k in range(1, 11):
        feats = comp_results[m]["features"][k-1]
        lr = LinearRegression()
        lr.fit(X_train_scaled[feats], y_train)
        preds = lr.predict(X_test_scaled[feats])
        comp_results[m]["rmse"].append(np.sqrt(mean_squared_error(y_test, preds)))
        comp_results[m]["r2"].append(r2_score(y_test, preds))

# Set up matplotlib figure with gridspec
fig = plt.figure(figsize=(16, 17))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])
ax_rmse = fig.add_subplot(gs[0, 0])
ax_r2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

plot_styles = {
    "Stepwise Manual": {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
    "Pearson Correlation": {"marker": "s", "linestyle": "-", "color": "#ff7f0e"},
    "RFE (Recursive)": {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
    "Lasso L1": {"marker": "d", "linestyle": ":", "color": "#d62728"},
    "Random Forest": {"marker": "x", "linestyle": "-", "color": "#9467bd"}
}

# Plot RMSE
ax_rmse.set_title("RMSE by Feature Selection Schemes", fontsize=12, fontweight='bold', pad=10)
for m in methods:
    ax_rmse.plot(
        range(1, 11), 
        comp_results[m]["rmse"], 
        label=m,
        marker=plot_styles[m]["marker"],
        linestyle=plot_styles[m]["linestyle"],
        color=plot_styles[m]["color"],
        markersize=7,
        linewidth=1.8
    )
ax_rmse.set_xlabel("Number of Features (k)", fontsize=10)
ax_rmse.set_ylabel("RMSE (Lower is Better)", fontsize=10)
ax_rmse.set_xticks(range(1, 11))
ax_rmse.grid(True, linestyle="--", alpha=0.5)
ax_rmse.legend(loc="upper right", fontsize=9)

# Plot R-squared
ax_r2.set_title("R-squared by Feature Selection Schemes", fontsize=12, fontweight='bold', pad=10)
for m in methods:
    ax_r2.plot(
        range(1, 11), 
        comp_results[m]["r2"], 
        label=m,
        marker=plot_styles[m]["marker"],
        linestyle=plot_styles[m]["linestyle"],
        color=plot_styles[m]["color"],
        markersize=7,
        linewidth=1.8
    )
ax_r2.set_xlabel("Number of Features (k)", fontsize=10)
ax_r2.set_ylabel("R-squared (Higher is Better)", fontsize=10)
ax_r2.set_xticks(range(1, 11))
ax_r2.grid(True, linestyle="--", alpha=0.5)
ax_r2.legend(loc="lower right", fontsize=9)

# Prepare table data
table_data = []
for k in range(1, 11):
    row = [str(k)]
    for m in methods:
        feats = comp_results[m]["features"][k-1]
        # Wrap features every 5 items with newlines to prevent text overlap in column cells
        wrapped = "\n".join([", ".join(feats[i:i+5]) for i in range(0, len(feats), 5)])
        row.append(wrapped)
    table_data.append(row)

columns = ["k"] + methods
ax_table.axis('off')
table = ax_table.table(
    cellText=table_data,
    colLabels=columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e6e6e6')
    else:
        if col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')

plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.06, hspace=0.22, wspace=0.18)
plt.savefig("feature_selection_comparison.png", dpi=200)
plt.close()

print("All tasks finished successfully.")
