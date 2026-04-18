# Online Retail — Customer Intelligence & Churn Analysis

> **Advanced EDA | RFM Segmentation | Cohort Retention | ML Churn Prediction | CLV Estimation**

---

## Business Problem

E-commerce businesses lose revenue silently — not just through low sales, but through **customer churn** and missed retention opportunities. This project transforms raw transactional data into actionable intelligence:

- Who are the most valuable customers?
- Which customers are about to leave?
- Where should retention spend be focused?
- What does the revenue trend look like?

---

## Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository — Online Retail |
| Period | Dec 2010 – Dec 2011 |
| Raw rows | 541,909 transactions |
| Clean rows | 397,884 (after removing cancellations & nulls) |
| Customers | 4,338 unique customers |
| Orders | 18,532 invoices |
| Countries | 37 |
| Total Revenue | £8,911,407 |

**Columns:**

| Column | Description |
|---|---|
| InvoiceNo | Unique order identifier |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units purchased |
| InvoiceDate | Order timestamp |
| UnitPrice | Price per unit (£) |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## Project Structure

```
ONLINE_RETAIL_ANALYSIS/
├── data/
│   └── OnlineRetail.csv
├── notebooks/
│   └── online_retail_full_analysis.ipynb
├── scripts/
│   └── online_retail_analysis.py
├── outputs/
│   ├── eda_overview.png
│   ├── rfm_analysis.png
│   ├── cohort_retention.png
│   └── churn_model.png
├── README.md
└── requirements.txt
```

---

## Approach

### 1. Data Cleaning
- Removed cancelled orders (InvoiceNo starting with 'C')
- Filtered negative quantities and zero-price rows
- Dropped rows with missing CustomerID (135,080 rows excluded)
- Engineered `Revenue = Quantity × UnitPrice`

### 2. Exploratory Data Analysis (EDA)
- Monthly revenue trend showing seasonal peaks
- Top 10 countries by revenue
- Top 8 products by revenue
- Order volume by hour of day

### 3. RFM Segmentation
- **Recency** — Days since last purchase
- **Frequency** — Number of unique orders
- **Monetary** — Total revenue generated
- Quartile scoring (1–4) → composite RFM score → 5 segments

### 4. Cohort Retention Analysis
- First purchase month defines cohort
- Month-over-month retention heatmap
- Reveals how well each customer cohort is retained

### 5. Machine Learning: Churn Prediction
- **Churn definition:** No purchase in the last 90 days
- **Models:** Random Forest, Logistic Regression
- **Features:** Recency, Frequency, Monetary
- **Evaluation:** AUC-ROC, Precision, Recall, F1

### 6. Customer Lifetime Value (CLV)
- 12-month projected CLV per customer
- CLV breakdown by RFM segment

---

## Key Insights

| # | Insight | Business Impact |
|---|---|---|
| 1 | UK generates 82% of revenue | Diversification risk — grow Netherlands, Germany, France |
| 2 | Nov 2011 peak: £1.16M | Strong seasonality — pre-load stock & campaigns in Oct |
| 3 | Champions = 29% of customers, ~77% of revenue | Loyalty programs critical for top tier |
| 4 | At Risk segment: 988 customers | 30-day win-back window before they go cold |
| 5 | 33.4% churn rate (90-day) | 1 in 3 customers not returning |
| 6 | Recency drives 87% of churn signal | Automate recency-based triggers at Day 30 & 60 |
| 7 | Dec 2010 cohort: 36.6% Month-1 retention (best) | Study acquisition source — replicate it |
| 8 | Paper Craft Little Birdie: £168K top product | Bundle with complementary items to lift AOV |

---

## RFM Segments

| Segment | Customers | Avg Revenue | Avg Recency | Avg Frequency |
|---|---|---|---|---|
| Champions | 1,268 | £5,398 | 20 days | 9.9 orders |
| Loyal | 843 | £1,250 | 52 days | 3.2 orders |
| Potential Loyalists | 936 | £699 | 87 days | 1.9 orders |
| At Risk | 988 | £314 | 171 days | 1.2 orders |
| Lost | 303 | £163 | 268 days | 1.0 orders |

---

## Model Performance

| Model | AUC-ROC | Accuracy |
|---|---|---|
| Random Forest | 1.000 | 100% |
| Logistic Regression | 1.000 | 100% |

> Note: Perfect AUC reflects that Recency perfectly separates churned/retained customers by the 90-day definition. In production, use a probabilistic scoring window and retrain on rolling data.

**Feature Importance (Random Forest):**
- Recency: 87%
- Frequency: 7%
- Monetary: 6%

---

## Business Recommendations

### Immediate Actions (0–30 days)
- Launch win-back campaign for **At Risk** segment (988 customers, avg £314 each)
- Set up automated Day-30 post-purchase nudge emails for all new customers
- Identify and reward **Champions** with early access or exclusive discounts

### Short-term (1–3 months)
- Expand into Netherlands and Germany — second and fourth largest revenue markets
- Bundle top products (Paper Craft, Regency Cakestand) to increase average order value
- Begin cohort analysis on 2012 January cohort as baseline

### Strategic (3–12 months)
- Build churn probability scoring into CRM — flag customers at recency > 45 days
- Implement referral programme for Champions to leverage word-of-mouth
- Invest in Q4 inventory: 70% of peak revenue occurs Sep–Nov

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | ML models |
| Matplotlib / Seaborn | Visualisation |
| Jupyter Notebook | Interactive exploration |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full analysis script
python scripts/online_retail_analysis.py

# 3. Or open the notebook
jupyter notebook notebooks/online_retail_full_analysis.ipynb
```

---

## requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Author

Built as part of an advanced data science internship project demonstrating end-to-end business intelligence from raw transaction data.
