# ✈️ TravelTide: Customer Personalization & Uplift Modeling

## 📌 Overview

TravelTide is a fast-growing e-booking startup that offers one of the largest travel inventories in the market. While the platform excels in search and availability, customer retention remains a key challenge.

This project focuses on designing a **data-driven, personalized rewards strategy** to increase customer engagement, conversion, and long-term value.

---

## 🎯 Business Problem

The marketing team, led by Elena Tarrant (Head of Marketing), aims to launch a **personalized rewards program**.

Instead of sending generic offers, the goal is to:

> Identify the most effective perk for each customer and maximize the probability of booking.

However, incentives (e.g., discounts) come with a cost, so the strategy must also **optimize Return on Investment (ROI)**.

---

## 🧠 Solution Approach

This project combines **analytics, machine learning, and business logic**:

### 1. Feature Engineering (Python)

Customer-level features were created from behavioral data:

* Session duration
* Booking behavior (flight, hotel, bundled trips)
* Discount usage
* Demographics (age, family status)

---

### 2. Uplift Modeling (T-Learner)

To estimate the **causal effect of discounts**, a two-model approach was used:

* Model A → predicts booking probability **with discount**
* Model B → predicts booking probability **without discount**

👉 Uplift is defined as:

[
\text{Uplift} = P(\text{Booking | Discount}) - P(\text{Booking | No Discount})
]

This allows us to identify **customers whose behavior is influenced by the discount**.

---

### 3. Customer Segmentation

Customers are segmented based on uplift:

* **Persuadable** → likely to respond to discount
* **Sure Thing** → will book anyway
* **Do Not Disturb** → negative impact from discount
* **Lost Cause** → unlikely to convert

---

### 4. ROI-Based Decision Engine

A decision framework was developed to optimize marketing spend:

[
\text{Expected Value} = (\text{Uplift} \times \text{Revenue}) - \text{Cost}
]

Customers are targeted only if the **expected value is positive**, ensuring efficient use of incentives.

---

### 5. Interactive Tableau Dashboard

An interactive dashboard was built in Tableau to simulate business scenarios:

* Adjustable parameters:

  * Cost of discount
  * Revenue per booking
* Real-time decision updates
* Customer segmentation visualization
* Uplift vs. booking probability analysis

---

## 📊 Key Insights

* Not all customers should receive discounts
* A large portion of users are **“Sure Things”** → discounts waste money
* Targeting only **high-uplift users** significantly improves ROI
* Personalization is critical for increasing conversion rates

---

## 🛠️ Tech Stack

* **Python**: pandas, numpy, scikit-learn
* **SQL**: data filtering and feature engineering
* **Tableau**: dashboard & decision visualization
* **GitHub**: version control and project documentation

---

## 📁 Project Structure

```
TravelTide_Project/
│
├── data/
│   └── merged_data.csv
│
│
├── src/
│   └── data_merging.py
│   └── feature_engineering.py
│   └── visualisation.py
│   └── logistic_regression.py
│   └── perk_signals.py
│   └── ML_Uplift_modeling.py
│
├── output/
│   └── uplift_decision.csv
│
├── dashboard/
│   └── tableau_dashboard.twbx
│
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/SoghraHajjar/traveltide-project.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the model:

```bash
python uplift_model.py
```

4. Open Tableau dashboard:

* Load `uplift_decision.csv`
* Adjust parameters to simulate decisions

---

## 💡 Key Takeaway

> Instead of giving discounts to all users, this project demonstrates how **uplift modeling + ROI optimization** can target only the customers who are truly influenced—maximizing conversions while minimizing cost.


---

## 👩‍💻 About Me

I’m a data analyst with a background in biostatistics, focused on applying statistical and machine learning methods to real-world problems.

This project reflects my ability to:

Think critically about data
Apply the right method for the right question
Bridge the gap between analysis and business decisions

