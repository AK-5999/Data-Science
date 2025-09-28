# 📊 EDA & Modeling on Zomato (Mumbai) Dataset — SQL + Python

## 🏷️ Project Title
**Exploratory Data Analysis and Predictive Modeling on Zomato (Mumbai) Dataset using SQL & Python**

---

## 🔍 Overview
This project demonstrates end-to-end exploratory data analysis (EDA) using **SQL** and predictive modeling using **Python** on a restaurants dataset (Zomato — Mumbai).  
Workflows include data ingestion into a relational DB, interactive SQL queries for EDA, feature engineering, and building simple predictive models (rating / cost / category) in Python to extract actionable business insights.

---

## 💼 Business Justification
- **Operational Insights**: Helps restaurants and aggregators understand price/rating patterns by locality and category.  
- **Data-Driven Decisions**: Assist marketing and menu/pricing strategies using empirical evidence.  
- **Scalable Analysis**: SQL-first approach allows exploration directly in database without heavy data transfer.  
- **Skill Relevance**: Demonstrates practical EDA with SQL and reproducible modeling in Python — highly relevant to analytics roles.

---

## 🧰 Tools & Technologies
- **Database / SQL**: MySQL / PostgreSQL, MySQL Workbench / PopSQL  
- **Python**: Jupyter Notebook, pandas, scikit-learn, XGBoost (optional)  
- **Visualization**: Matplotlib, Seaborn, Plotly (optional)  
- **Others**: Git, Excel (for small checks), PowerPoint (presentation)

---

## 🧾 Dataset Details
- **Source**: Zomato dataset (Kaggle / sample) — Mumbai restaurants  
- **Rows**: ~7,500 (example dataset size)  
- **Columns (example)**:
  - `name`, `locality`, `cuisines`, `average_cost_for_two`, `aggregate_rating`, `rating_text`, `votes`, `establishment`, `address`
- **Primary modeling targets (examples)**:
  - Predict `aggregate_rating` (regression)  
  - Classify `rating_text` (Good/Bad) or predict `average_cost_for_two` band (classification/regression)  

---

## 📥 How to Get Started
1. Clone the repo.  
2. Load CSV to MySQL (or SQLite/postgres). Example with MySQL Workbench: create database `zomato`, use Table Data Import Wizard.  
3. Open `notebooks/` in Jupyter and run the SQL queries + Python notebooks.

---

## 🔎 EDA — SQL Workflow & Sample Queries

### 1. Inspect data & size
```sql
-- Count rows & columns
SELECT COUNT(*) AS num_rows FROM zomato_data;
SELECT COUNT(*) AS num_columns
FROM information_schema.columns
WHERE table_name = 'zomato_data';
```
### 2. Basic aggregates
```sql
-- Average cost for two
SELECT ROUND(AVG(average_cost_for_two),2) AS avg_cost_for_2
FROM zomato_data;

-- Average rating
SELECT ROUND(AVG(aggregate_rating),2) AS avg_rating
FROM zomato_data;
```
### 3. Distribution & top entities
```sql
-- Unique localities count
SELECT COUNT(DISTINCT(locality)) AS num_localities FROM zomato_data;

-- Top 10 restaurant chains by outlet count
SELECT name, COUNT(*) AS outlets
FROM zomato_data
GROUP BY name
ORDER BY outlets DESC
LIMIT 10;

-- Most common establishment types
SELECT establishment, COUNT(*) AS count
FROM zomato_data
GROUP BY establishment
ORDER BY count DESC
LIMIT 10;
```
### 4. Category analysis
```sql
-- Average rating by establishment type
SELECT establishment, ROUND(AVG(aggregate_rating),2) AS avg_rating
FROM zomato_data
GROUP BY establishment
ORDER BY avg_rating DESC
LIMIT 10;
```
### 5. Locality & cuisine insight
```sql
-- Top cuisines in a locality (example: "Andheri")
SELECT cuisines, COUNT(*) AS cnt
FROM zomato_data
WHERE locality = 'Andheri'
GROUP BY cuisines
ORDER BY cnt DESC
LIMIT 10;
```
## 🧠 EDA — Notes & Tips
- Use GROUP BY, ORDER BY and window functions (if supported) to derive ranking and rolling stats.
- Clean inconsistent text fields (e.g., establishment, cuisines) using SQL string functions or in Python after extraction.
- Join with external locality data (latitude/longitude) if you want spatial analysis.
