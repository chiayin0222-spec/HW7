# 🏠 加州房價預測與多重特徵篩選分析 (California Housing Price Prediction & Feature Selection Analysis)

本專案是一個基於加州房價資料集（California Housing Dataset）的機器學習分析與預測專案。採用 **CRISP-DM（跨行業資料探勘標準流程）** 框架，透過詳細的資料探索（EDA）、資料清洗與預處理，並實作多達 **11 種不同的特徵選取（Feature Selection）機制**，最終使用多種迴歸模型（如隨機森林 Regressor 與線性迴歸）進行預測與性能評估。

---

## 📂 專案檔案結構 (Project Files Layout)

*   **主執行程式**：[`0615-01.py`](file:///d:/Hazel/Antig/0615pm/0615-01.py) — 核心 Python 腳本，負責執行資料載入、清洗、11 種特徵選取排名、模型評估，並輸出所有視覺化圖表與績效看板。
*   **詳細分析報告**：[`0615-01.md`](file:///d:/Hazel/Antig/0615pm/0615-01.md) — 系統化的專案報告，記錄了完整的 CRISP-DM 流程步驟與詳細數據。
*   **資料集檔案**：`date.csv` — 原始的加州房價資料集，共計 20,640 筆觀測值與 10 個屬性特徵。

### 📊 產出的視覺化分析圖表
本專案執行後會自動產生以下關鍵視覺化成果：
1.  **綜合分析看板 (`allinone.png`)**：包含相關性熱力圖、收入與房價關係散點圖、靠海程度盒鬚圖，以及前 10 大重要特徵長條圖的四合一看板。
2.  **特徵篩選方案對比圖 (`california_feature_selection_comparison.png`)**：展示 11 種篩選方案在特徵數量 $k \in [1, 10]$ 下的 RMSE 與 $R^2$ 走勢變化，並附帶特徵排序矩陣對照表。
3.  **單一圖表檔案**：
    *   `california_correlation_heatmap.png` — 欄位間的 Pearson 相關性係數熱力矩陣。
    *   `income_vs_house_value.png` — 探討中位數收入 (Median Income) 與中位數房價 (Median House Value) 的關係。
    *   `ocean_proximity_vs_house_value.png` — 地理靠海類別對房價的分布影響。
    *   `california_feature_importance.png` — 隨機森林演算法算出的特徵重要度權重排名。

---

## 🛠️ 環境建置與安裝 (Setup & Installation)

請確保您的 Python 環境已安裝以下套件：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 執行指南 (How to Run)

只需在專案目錄下執行 [`0615-01.py`](file:///d:/Hazel/Antig/0615pm/0615-01.py) 腳本，即可完成完整的流程並生成所有圖表：

```bash
python 0615-01.py
```

---

## 🔍 資料集特徵與預處理 (Data & Preprocessing)

### 資料規格 (Dataset Shape)
*   **樣本數**：20,640
*   **特徵數**：10

### 特徵字典 (Feature Dictionary)
| 欄位名稱 (Feature) | 說明 (Description) |
| :--- | :--- |
| `longitude` | 經度 |
| `latitude` | 緯度 |
| `housing_median_age` | 區域內房屋年齡中位數 |
| `total_rooms` | 區域內房間總數 |
| `total_bedrooms` | 區域內臥室總數 |
| `population` | 區域內總人口 |
| `households` | 區域內總戶數 |
| `median_income` | 區域內家庭年收入中位數（單位：萬美元） |
| `ocean_proximity` | 房屋與海岸的地理鄰近程度（類別特徵） |
| **`median_house_value`** | **目標變數 (Target)：區域房屋價值中位數** |

### 預處理步驟
1.  **缺失值填補**：發現 `total_bedrooms` 有約 207 筆缺失值，採用該欄位之**中位數 (Median)** 進行填補。
2.  **類別特徵編碼**：將非數值特徵 `ocean_proximity` 進行 **One-Hot Encoding** (使用 `drop_first=True` 避免共線性)。
3.  **特徵標準化**：將特徵矩陣透過 `StandardScaler` 標準化，使機器學習模型訓練更穩定。

---

## 🧮 11 種特徵選取機制 (Feature Selection Schemes)

為了篩選出預測加州房價最有效的特徵組合，本專案實作並比較了以下 **11 種特徵選取算法** 的前 10 大特徵排名：

1.  **Pearson Correlation** (皮爾森相關係數)
2.  **Spearman Correlation** (斯皮爾曼等級相關係數)
3.  **F-test** (線性模型顯著性考驗分數)
4.  **Mutual Information** (互資訊，捕捉非線性關係)
5.  **RFE** (遞迴特徵消除法，基於線性模型)
6.  **SFS** (循序前向選擇，基於交叉驗證 MSE)
7.  **SBS** (循序後向消除，基於交叉驗證 MSE)
8.  **Hill Climbing** (爬山演算法尋優特徵子集)
9.  **Genetic Algorithm** (遺傳演算法，模擬自然選擇演化特徵組合)
10. **Lasso (L1)** (套索迴歸係數收縮特性)
11. **Random Forest Importance** (隨機森林樹節點分裂不純度貢獻度)

---

## 📈 評估結果與主要發現 (Results & Insights)

### 1. 隨機森林模型評估結果
使用隨機森林迴歸器 (Random Forest Regressor) 訓練完整特徵，在測試集上得到極佳的擬合效果：
*   **平均絕對誤差 (MAE)**：31,470.222
*   **均方根誤差 (RMSE)**：48,769.615
*   **判定係數 ($R^2$ Score)**：**0.818** (可解釋測試集約 81.8% 的房價變異)

### 2. 關鍵影響因子分析
*   **中位數收入 (median_income)**：不論在何種特徵選取方案中，中位數收入皆被評為**最重要特徵**。散點圖顯示收入與房價有強大的正向線性關聯。
*   **地理位置 (latitude / longitude)**：加州的地理特徵具有非常明顯的地域溢價（如舊金山灣區與洛杉磯的房價高峰）。
*   **靠海程度 (ocean_proximity)**：內陸地區 (`INLAND`) 對房價有強烈的負面作用（相對價格較低），而臨海住宅則具有高度溢價空間。

---

## 💡 商業建議 (Business Recommendations)

*   **房地產開發與選址**：開發商應優先挑選高收入人口聚集區與沿海地帶進行新案開發；在內陸地區則建議以主打高性價比的剛性需求住宅為主。
*   **房產投資決策**：投資者應特別關注家庭中位數收入持續成長、且具備良好地理區位或自然景觀（如靠海）的社區，這些特徵為房價提供了極佳的支撐與升值潛力。
