import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE, mutual_info_regression, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load California Housing Dataset
print("Loading California Housing Dataset from date.csv...")
df = pd.read_csv("date.csv")

# Print Dataset Shape
print(f"Dataset Shape: {df.shape}")

# Fill Missing Values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Encode Categorical Feature
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Split train/test data
X = df_encoded.drop("median_house_value", axis=1)
y = df_encoded["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Define 11 Feature Selection Schemes for Ranking
print("\nRunning 11 feature selection ranking schemes...")

# 1. Pearson Correlation
pearson_rank = df_encoded.corr(method="pearson")["median_house_value"].abs().drop("median_house_value").sort_values(ascending=False).index[:10].tolist()

# 2. Spearman Correlation
spearman_rank = df_encoded.corr(method="spearman")["median_house_value"].abs().drop("median_house_value").sort_values(ascending=False).index[:10].tolist()

# 3. F-test
f_scores, _ = f_regression(X_train_scaled, y_train)
f_test_rank = pd.Series(f_scores, index=X_train_scaled.columns).sort_values(ascending=False).index[:10].tolist()

# 4. Mutual Info
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_rank = pd.Series(mi_scores, index=X_train_scaled.columns).sort_values(ascending=False).index[:10].tolist()

# 5. RFE (Recursive Feature Elimination)
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=1, step=1)
selector.fit(X_train_scaled, y_train)
rfe_rank = [X_train_scaled.columns[i] for i in np.argsort(selector.ranking_)][:10]

# 6. SFS (Forward Selection)
def get_sfs_features(X_tr, y_tr, k):
    selected = []
    remaining = list(X_tr.columns)
    for _ in range(k):
        best_score = float('inf')
        best_feat = None
        for feat in remaining:
            candidate = selected + [feat]
            model = LinearRegression()
            scores = cross_val_score(model, X_tr[candidate], y_tr, cv=3, scoring='neg_mean_squared_error')
            score = -scores.mean()
            if score < best_score:
                best_score = score
                best_feat = feat
        if best_feat:
            selected.append(best_feat)
            remaining.remove(best_feat)
    return selected

print("Running SFS (Forward)...")
sfs_rank = get_sfs_features(X_train_scaled, y_train, 10)

# 7. SBS (Backward Selection)
def get_sbs_features(X_tr, y_tr, k):
    selected = list(X_tr.columns)
    eliminated_order = []
    while len(selected) > 1:
        worst_score = float('inf')
        worst_feat = None
        for feat in selected:
            candidate = [f for f in selected if f != feat]
            model = LinearRegression()
            scores = cross_val_score(model, X_tr[candidate], y_tr, cv=3, scoring='neg_mean_squared_error')
            score = -scores.mean()
            if score < worst_score:
                worst_score = score
                worst_feat = feat
        if worst_feat:
            selected.remove(worst_feat)
            eliminated_order.append(worst_feat)
    eliminated_order.append(selected[0])
    return eliminated_order[::-1][:k]

print("Running SBS (Backward)...")
sbs_rank = get_sbs_features(X_train_scaled, y_train, 10)

# 8. Hill Climbing (爬山演算法)
def get_hill_climbing_ranking(X_tr, y_tr, num_iter=50):
    n_features = X_tr.shape[1]
    current_mask = np.random.choice([True, False], size=n_features)
    
    def evaluate_mask(mask):
        if not any(mask):
            return float('inf')
        cols = [X_tr.columns[i] for i, val in enumerate(mask) if val]
        model = LinearRegression()
        scores = cross_val_score(model, X_tr[cols], y_tr, cv=3, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    current_score = evaluate_mask(current_mask)
    feature_counts = np.zeros(n_features)
    
    for _ in range(num_iter):
        flip_idx = np.random.randint(0, n_features)
        new_mask = current_mask.copy()
        new_mask[flip_idx] = not new_mask[flip_idx]
        new_score = evaluate_mask(new_mask)
        if new_score < current_score:
            current_mask = new_mask
            current_score = new_score
        feature_counts += current_mask
        
    scores_series = pd.Series(feature_counts, index=X_tr.columns)
    return scores_series.sort_values(ascending=False).index[:10].tolist()

print("Running Hill Climbing...")
hc_rank = get_hill_climbing_ranking(X_train_scaled, y_train, 50)

# 9. Genetic Algorithm (遺傳演算法)
def get_genetic_algorithm_ranking(X_tr, y_tr, pop_size=10, generations=5, mutation_rate=0.1):
    n_features = X_tr.shape[1]
    population = np.random.choice([0, 1], size=(pop_size, n_features))
    
    def fitness(chromosome):
        if sum(chromosome) == 0:
            return -float('inf')
        cols = [X_tr.columns[i] for i, val in enumerate(chromosome) if val == 1]
        model = LinearRegression()
        scores = cross_val_score(model, X_tr[cols], y_tr, cv=3, scoring='neg_mean_squared_error')
        return scores.mean()
    
    for gen in range(generations):
        fit_scores = np.array([fitness(chrom) for chrom in population])
        if np.all(fit_scores == -float('inf')):
            fit_scores = np.zeros(pop_size)
        ranks = np.argsort(np.argsort(fit_scores))
        probs = (ranks + 1) / (ranks + 1).sum()
        idx_parents = np.random.choice(range(pop_size), size=pop_size, p=probs)
        parents = population[idx_parents]
        next_pop = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i+1, pop_size-1)]
            cross_pt = np.random.randint(1, n_features)
            c1 = np.concatenate([p1[:cross_pt], p2[cross_pt:]])
            c2 = np.concatenate([p2[:cross_pt], p1[cross_pt:]])
            next_pop.extend([c1, c2])
        population = np.array(next_pop)[:pop_size]
        for chrom in population:
            for gene_idx in range(n_features):
                if np.random.rand() < mutation_rate:
                    chrom[gene_idx] = 1 - chrom[gene_idx]
                    
    fit_scores = np.array([fitness(chrom) for chrom in population])
    ranks = np.argsort(np.argsort(fit_scores))
    feature_scores = np.zeros(n_features)
    for i, chrom in enumerate(population):
        feature_scores += chrom * (ranks[i] + 1)
        
    scores_series = pd.Series(feature_scores, index=X_tr.columns)
    return scores_series.sort_values(ascending=False).index[:10].tolist()

print("Running Genetic Algorithm...")
ga_rank = get_genetic_algorithm_ranking(X_train_scaled, y_train, 10, 5, 0.1)

# 10. Lasso (L1)
lasso = LassoCV(cv=3, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
lasso_rank = pd.Series(np.abs(lasso.coef_), index=X_train_scaled.columns).sort_values(ascending=False).index[:10].tolist()

# 11. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_rank = pd.Series(rf_model.feature_importances_, index=X_train_scaled.columns).sort_values(ascending=False).index[:10].tolist()

methods = [
    "Pearson", "Spearman", "F-test", "Mutual Info", 
    "RFE", "SFS (Fwd)", "SBS (Bwd)", "Hill Climb",
    "Genetic Algo", "Lasso (L1)", "Random Forest"
]

rank_lists = {
    "Pearson": pearson_rank,
    "Spearman": spearman_rank,
    "F-test": f_test_rank,
    "Mutual Info": mi_rank,
    "RFE": rfe_rank,
    "SFS (Fwd)": sfs_rank,
    "SBS (Bwd)": sbs_rank,
    "Hill Climb": hc_rank,
    "Genetic Algo": ga_rank,
    "Lasso (L1)": lasso_rank,
    "Random Forest": rf_rank
}

# Evaluate performance metrics for each method across k=1 to 10
print("\nEvaluating metrics across k=1 to 10 for each scheme...")
results = {m: {"rmse": [], "r2": []} for m in methods}
for m in methods:
    for k in range(1, 11):
        feats = rank_lists[m][:k]
        lr = LinearRegression()
        lr.fit(X_train_scaled[feats], y_train)
        preds = lr.predict(X_test_scaled[feats])
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results[m]["rmse"].append(rmse)
        results[m]["r2"].append(r2)

# Set up matplotlib figure with gridspec
print("Plotting comparison dashboard...")
fig = plt.figure(figsize=(26, 14))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
ax_rmse = fig.add_subplot(gs[0, 0])
ax_r2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])

plot_styles = {
    "Pearson": {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
    "Spearman": {"marker": "s", "linestyle": "-", "color": "#aec7e8"},
    "F-test": {"marker": "^", "linestyle": "-.", "color": "#ff7f0e"},
    "Mutual Info": {"marker": "d", "linestyle": ":", "color": "#ffbb78"},
    "RFE": {"marker": "v", "linestyle": "--", "color": "#2ca02c"},
    "SFS (Fwd)": {"marker": "x", "linestyle": "-", "color": "#98df8a"},
    "SBS (Bwd)": {"marker": "*", "linestyle": "--", "color": "#d62728"},
    "Hill Climb": {"marker": "D", "linestyle": "-.", "color": "#ff9896"},
    "Genetic Algo": {"marker": "h", "linestyle": ":", "color": "#9467bd"},
    "Lasso (L1)": {"marker": "p", "linestyle": "-", "color": "#c5b0d5"},
    "Random Forest": {"marker": "+", "linestyle": "-.", "color": "#8c564b"}
}

# Plot RMSE
ax_rmse.set_title("RMSE by Feature Selection Schemes", fontsize=12, fontweight='bold', pad=10)
for m in methods:
    ax_rmse.plot(
        range(1, 11), 
        results[m]["rmse"], 
        label=m,
        marker=plot_styles[m]["marker"],
        linestyle=plot_styles[m]["linestyle"],
        color=plot_styles[m]["color"],
        markersize=6,
        linewidth=1.5
    )
ax_rmse.set_xlabel("Number of Features (k)", fontsize=10)
ax_rmse.set_ylabel("RMSE (Lower is Better)", fontsize=10)
ax_rmse.set_xticks(range(1, 11))
ax_rmse.grid(True, linestyle="--", alpha=0.5)
ax_rmse.legend(loc="upper right", fontsize=8.0, ncol=2)

# Plot R-squared
ax_r2.set_title("R-squared by Feature Selection Schemes", fontsize=12, fontweight='bold', pad=10)
for m in methods:
    ax_r2.plot(
        range(1, 11), 
        results[m]["r2"], 
        label=m,
        marker=plot_styles[m]["marker"],
        linestyle=plot_styles[m]["linestyle"],
        color=plot_styles[m]["color"],
        markersize=6,
        linewidth=1.5
    )
ax_r2.set_xlabel("Number of Features (k)", fontsize=10)
ax_r2.set_ylabel("R-squared (Higher is Better)", fontsize=10)
ax_r2.set_xticks(range(1, 11))
ax_r2.grid(True, linestyle="--", alpha=0.5)
ax_r2.legend(loc="lower right", fontsize=8.0, ncol=2)

# Prepare Rank Table data
table_data = []
for k in range(1, 11):
    row = [f"Rank {k}"]
    for m in methods:
        row.append(rank_lists[m][k-1])
    table_data.append(row)

columns = ["Rank"] + methods
ax_table.axis('off')
table = ax_table.table(
    cellText=table_data,
    colLabels=columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(7.5)

# Set uniform heights and headers for Rank Table
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1f3a60')
        cell.set_height(0.06)
    else:
        cell.set_height(0.05)
        if col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
        else:
            cell.set_facecolor('#fdfdfd')

plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.05, hspace=0.25, wspace=0.18)
plt.savefig("california_feature_selection_comparison.png", dpi=200)
plt.close()

# Also run individual plots and save for completeness
print("Generating standard California Housing EDA and model...")
plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("california_correlation_heatmap.png", dpi=200)
plt.close()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.4)
plt.title("Income vs House Value")
plt.tight_layout()
plt.savefig("income_vs_house_value.png", dpi=200)
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value")
plt.title("Ocean Proximity vs House Value")
plt.tight_layout()
plt.savefig("ocean_proximity_vs_house_value.png", dpi=200)
plt.close()

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance.head(10), x="Importance", y="Feature", hue="Feature", legend=False, palette="viridis")
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("california_feature_importance.png", dpi=200)
plt.close()

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R^2  : {r2:.3f}")

# Generate Combined Dashboard
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0, 0], annot_kws={"size": 8})
axes[0, 0].set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.4, ax=axes[0, 1])
axes[0, 1].set_title("Income vs House Value", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Median Income")
axes[0, 1].set_ylabel("Median House Value")
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", ax=axes[1, 0])
axes[1, 0].set_title("Ocean Proximity vs House Value", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Ocean Proximity")
axes[1, 0].set_ylabel("Median House Value")
sns.barplot(data=importance.head(10), x="Importance", y="Feature", palette="viridis", ax=axes[1, 1], hue="Feature", legend=False)
axes[1, 1].set_title("Top 10 Important Features (Random Forest)", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Importance")
axes[1, 1].set_ylabel("Feature")
plt.suptitle("California Housing Data Analysis Dashboard", fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("allinone.png", dpi=200)
plt.close()

print("\nFinished successfully.")
