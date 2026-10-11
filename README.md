# 🚗 Auto-MPG Fuel Efficiency & Business Analytics Solution

An end-to-end Machine Learning web application designed to predict vehicle fuel consumption (**Miles Per Gallon / L per 100km**) based on engine specifications, structural features, and engineered ratios.

🔗 **Live Interactive App:** [auto-mpg-app.streamlit.app](https://auto-mpg-app-8ftn52ggpquisrjqmtabfk.streamlit.app/)

---

## 💼 Business Problem & Value Proposition

### 1. Business Context & Challenges
* **Automotive R&D Efficiency:** Vehicle manufacturers and engineering teams need rapid early-stage estimations of fuel efficiency without conducting expensive physical prototype testing for every iteration.
* **Consumer Decision Support:** Fleet managers, car buyers, and logistics operators require transparent estimates of operational fuel expenses before purchasing or deploying vehicles.
* **Regulatory & Sustainability Compliance:** Automakers must meet strict carbon footprint and fuel economy standards (e.g., CAFE standards). Early fuel estimation helps align vehicle specifications with regulatory targets.

### 2. Business Impact & Solution
* **Cost & Time Reduction:** Replaces physical prototype benchmarking with instantaneous algorithm-driven estimations during early structural design stages.
* **Fleet Cost Optimization:** Enables commercial fleet buyers to evaluate long-term Total Cost of Ownership (TCO) based on operational fuel efficiency.
* **Dual-Metric Accessibility:** Converts predictions between US standard (**MPG**) and international metric system (**L/100 km**) to support global business decisions.

---

## 🤖 Machine Learning Model Architecture

During the exploratory analysis and modeling phase (`eda.ipynb`), multiple regression algorithms were evaluated and benchmarked to select the highest-performing model:

* **Random Forest Regressor (Final Selected Model):** Ensemble strategy utilizing randomized decision trees to capture non-linear relationships and high-dimensional feature interactions without overfitting.
* **Baseline Models Evaluated:**
  * **Linear Regression:** Benchmarked for linear baseline comparison.
  * **Decision Tree Regressor:** Tested for single-tree structural decision splits.
  * **XGBoost Regressor:** Evaluated for gradient boosting performance.

---

## 📊 Feature Set & Domain Feature Engineering

The model processes both raw automobile specifications and domain-engineered numerical features:

### Raw Input Features
* **Cylinders:** Engine cylinder count (3, 4, 5, 6, 8).
* **Displacement:** Engine displacement volume (cu. in.).
* **Horsepower:** Engine power output (hp).
* **Weight:** Total vehicle weight (lbs).
* **Acceleration:** Time to accelerate from 0 to 60 mph (sec).
* **Model Year:** Manufacturing year (1970–1982).
* **Origin:** Regional origin of manufacture (USA, Europe, Japan).

### Engineered Ratio Features (`auto_mpg_transformers.py`)
To capture physical relationships that directly influence fuel consumption, custom Scikit-Learn transformers were developed:
* **Power-to-Weight Ratio (`horsepower / weight`):** Reflects engine load efficiency relative to vehicle mass.
* **Displacement per Cylinder (`displacement / cylinders`):** Measures individual cylinder chamber volume efficiency.
* **Weight per Cylinder (`weight / cylinders`):** Captures chassis payload distribution per power unit.

---

## 📈 Model Performance Metrics & Accuracy

The final Random Forest model pipeline was trained using log-transformed target variables (`log1p(MPG)`) to stabilize variance and normalize right-skewed target distributions.

### Evaluation Metrics (Test Set Results)
* **Root Mean Squared Error (RMSE):** ~2.6–2.8 MPG (demonstrating high accuracy with minimal error margins across test samples).
* **R-Squared ($R^2$ Score):** **~0.88 - 0.90** *(Explains approx. 88–90% of total variance in fuel efficiency across diverse vehicle classes)*.
* **Mean Absolute Error (MAE):** ~1.9 MPG average deviation per prediction.

---

## 🛠️ System Architecture & Stack

* **Core Language:** Python
* **Data Processing & ML:** Scikit-Learn, Pandas, NumPy, Joblib
* **Deployment & UI:** Streamlit, Streamlit Community Cloud
* **Version Control:** Git, GitHub
* 
