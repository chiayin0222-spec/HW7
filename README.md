# 🏠 房價預測與特徵選取分析專案 (Housing Price Prediction & Feature Selection Analysis)

本專案包含兩個獨立且完整的房價預測與機器學習工作流，採用 CRISP-DM 流程，分別針對 **波士頓房價 (Boston Housing)** 與 **加州房價 (California Housing)** 進行資料探索 (EDA)、資料預處理、多種特徵選取方案評估，以及迴歸模型訓練與評估。

---

## 📂 專案結構 (Project Directory Layout)

以下為專案的主要檔案與其功能說明：

### 1. 波士頓房價預測 (Boston Housing Price Prediction)
*   **主執行程式**：[`0615.py`](file:///d:/Hazel/Antig/0615pm/0615.py) - 下載資料集、訓練 10 種模型、比較 5 種特徵選擇方案並儲存分析圖表。
*   **詳細分析報告**：[`0615.md`](file:///d:/Hazel/Antig/0615pm/0615.md) - 詳細記錄 CRISP-DM 流程的各個階段與預測表現。
*   **分析圖表**：
    *   `correlation_heatmap.png`：特徵相關性熱力圖。
    *   `rm_vs_medv.png` / `lstat_vs_medv.png`：關鍵特徵 RM (房間數) 與 LSTAT (低收入人口比例) 對房價的散點圖。
    *   `top10_linear_curves.png`：前 10 大特徵與房價的線性擬合曲線圖。
    *   `feature_importance.png`：基於隨機森林 (Random Forest) 的特徵重要性排名。
    *   `models_r2_comparison.png`：10 種迴歸演算法的 $R^2$ 評估分數對比。
    *   `feature_selection_comparison.png`：5 種特徵篩選方案（Stepwise、Pearson、RFE、Lasso L1、隨機森林）在不同特徵數 $k$ 下的表現對比。

### 2. 加州房價預測 (California Housing Price Prediction)
*   **主執行程式**：[`0615-01.py`](file:///d:/Hazel/Antig/0615pm/0615-01.py) - 載入加州房價資料集、進行預處理、執行 11 種特徵篩選機制並儲存分析圖表。
*   **詳細分析報告**：[`0615-01.md`](file:///d:/Hazel/Antig/0615pm/0615-01.md) - 詳細記錄加州房價分析結果與特徵工程步驟。
*   **資料集**：`date.csv`（包含加州房屋資料，如經緯度、房間數、人口、收入及靠海程度等）
*   **分析圖表**：
    *   `allinone.png`：加州房價綜合資料探索 (EDA) 看板。
    *   `california_correlation_heatmap.png`：特徵相關性熱力圖。
    *   `income_vs_house_value.png`：中位數收入與房價的關聯散點圖。
    *   `ocean_proximity_vs_house_value.png`：房屋靠海程度與房價的盒鬚圖。
    *   `california_feature_importance.png`：隨機森林特徵重要性排行。
    *   `california_feature_selection_comparison.png`：11 種特徵篩選方案在不同特徵數 $k$ 下的性能指標 (RMSE & $R^2$) 及特徵排名表。

### 3. 其他檔案
*   [`claude.md`](file:///d:/Hazel/Antig/0615pm/claude.md) / `L10 2330/claude.md`：開發行為與編碼準則規章。

---

## 🛠️ 環境建置與安裝 (Setup & Installation)

執行本專案前，請確保您的 Python 環境（建議使用 Python 3.8+）已安裝以下套件：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 執行指南 (How to Run)

### 1. 執行波士頓房價分析
執行 `0615.py`，此程式會自動下載波士頓房價資料集，進行特徵工程與模型訓練，並於當前目錄下生成相關視覺化圖表。
```bash
python 0615.py
```

### 2. 執行加州房價分析
執行 `0615-01.py`，此程式會載入本地的 `date.csv` 資料集，進行缺失值填補、One-Hot 編碼與標準化，接著執行 11 種特徵選擇方法與預測評估，並輸出對應分析看板。
```bash
python 0615-01.py
```

---

## 📊 核心分析與模型表現摘要 (Key Insights & Model Performance)

### 1. 波士頓房價分析 (Boston Housing)
*   **模型表現比較**（10種演算法）：
    *   **最佳表現**：**Gradient Boosting** ($R^2 \approx 0.914$) 與 **XGBoost** ($R^2 \approx 0.905$) 表現最佳，大幅領先傳統線性迴歸 ($R^2 \approx 0.669$)。
*   **主要影響因素**：
    1.  **RM** (平均房間數)：呈顯著正相關，房間數越多房價越高。
    2.  **LSTAT** (低收入人口比例)：呈顯著負相關，比例越高的地區房價越低。
    3.  **NOX** (一氧化氮濃度)：空氣污染程度較高地區，房價偏低。

### 2. 加州房價分析 (California Housing)
*   **資料預處理**：針對 `total_bedrooms` 缺失值以中位數進行填補，並對 `ocean_proximity` 進行 Dummy 變數編碼。
*   **主要影響因素**：
    1.  **median_income** (區域中位數收入)：為最關鍵因子，高收入社區房價明顯較高。
    2.  **ocean_proximity** (靠海程度)：沿海與海景住宅相較於內陸地區 (Inland) 有顯著溢價效應。
    3.  **latitude/longitude** (經緯度地理位置)：加州各大都會區（如舊金山、洛杉磯）的地理特徵，對房價有著決定性影響。
*   **特徵篩選機制對比**（包含 Pearson、Spearman、F-test、Mutual Info、RFE、SFS、SBS、Hill Climbing、Genetic Algorithm、Lasso、Random Forest 共 11 種方案）：
    *   藉由評估不同特徵子集 $k \in [1, 10]$ 對線性迴歸的預測表現，可以看出各方案特徵組合的篩選效率，詳細表現收錄於 `california_feature_selection_comparison.png` 中的圖表與對照表。

---

## 💡 商業建議與決策參考 (Business Recommendations)

*   **不動產開發商與仲介商**：應優先在「高收入人口聚集區」與「沿海/成熟生活機能區」進行重點開發與房屋銷售，以獲取更高的溢價空間。
*   **投資者**：應避開空氣污染高 (NOX)、低收入比例增加 (LSTAT) 或犯罪率高 (CRIM) 的區域；並特別關注收入成長且地理交通便利之潛力新興區。
