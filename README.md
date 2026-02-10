# Application Fraud: Bank Account Opening Fraud Detection using Logistic Regression and XGBoost

**Date**: 9 February 2026 \
**Author**: Li Zhaozhi (李兆智)

---

**Background**: Fraud is a type of financial crime risk that poses threats to customers and banks. There're multiple typologies within fraud such as authorised and unauthorised digital, payment, credit card, application, scams, and lending fraud, etc. Financial institutions deploy data science capabilities in banking data to analyse fraud patterns, detect, and mitigate fraud risks.

**Application Fraud, Bank Account Opening**: refers to the deliberate submission of false, forged, or stolen information during the account opening process with the intent to:

- Obtain financial products/services under false pretenses
- Facilitate money laundering or other financial crimes
- Circumvent regulatory controls and due diligence requirements
- Create vehicles for future illicit activities

(Reference: Association of Certified Anti-Money Laundering Specialists (ACAMS))

**Objective**: This solution aims to discern patterns and detect application fraud in the form of bank account opening by applying statistical analysis, data visualisation, hypothesis testing, and machine learning techniques (Logistic Regression and XGBoost) to analyse and model bank account data. This covers the entire (iterative) life cycle from exploratory data analysis (EDA), data cleaning, hypothesis testing, feature engineering, modelling to model evaluation.

**Challenges**: Financial institutions face the following challenges in fraud detection:

- *False positive*: Genuine customers/transactions flagged as fraudulent leading to increased investigation expenses.
- *False negative*: Failure to detect fraudulent customers/transactions leading to financial loss and reputational damage, sometimes regulatory fines.
- *Class imblance*: Fraud data is typically imbalanced which requires processing before modelling.

**Methodologies**: To answer the above research questions, *Logistic Regression* and *XGBoost* are appropriate. In a regulated environment like banking, explainability is critical from model governance perspective. 

- *Logistic Regression* is a surpervised learning method that is highly transparent. This dataset has a binary label column `fraud_bool` with large quantities of numerical and categorical variables available for featuring engineering. Logistic regression is an appropriate method for predicting fraud.
- *XGBoost*: 

---

**Data Sets**: Feedzai is an AI-powered platform that uses machine learning to detect fraud. Feedzai Research released anonymised [data sets](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data) at NeurIPS 2022 resembling challenges in real-world bank account opening data. These data sets are available in downloadable CSV format. In real world, data is typically stored on a cloud platform and retrieved with SQL queries.

References:

- [Bank Account Fraud Dataset Suite Datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)

Data definitions:

| Num | Variable | Definition | Data Type | Unit | Example |
|--|----------|------------|-----------|------|---------|
|1|fraud_bool|Fraud label (1: Fraud, 0: geunine)|Numerical|N/A|1|
|2|income|Annual income in quantiles|Numerical|N/A|0.3|
|3|name_email_similarity|Metric of similarity between email and applicant’s name. Higher values represent higher similarity. Ranges between [0, 1].|Numerical|N/A|1|
|4|prev_address_months_count| Number of months in previous registered address of the applicant, i.e. the applicant’s previous residence, if applicable. Ranges between [−1, 380] months (-1 is a missing value).|Numerical|Month|2|
|5| current_address_months_count | Months in currently registered address of the applicant. Ranges between [−1, 406] months (-1 is a missing value). |Numerical|Month|100|
|6| customer_age | Applicant’s age in bins per decade (e.g, 20-29 is represented as 20). |Numerical|N/A|30|
|7|days_since_request|Number of days passed since application was done. Ranges between [0, 78] days.|Numerical|Day|12|
|8|intended_balcon_amount|Initial transferred amount for application. Ranges between [−1, 108].|Numerical|USD|100|
|9|payment_type|Credit payment plan type. 5 possible (annonymized) values.|Categorical|N/A|AD|
|10|zip_count_4w|Number of applications within same zip code in last 4 weeks. Ranges between [1, 5767].|Numerical|App|21|
|11|velocity_6h|Velocity of total applications made in last 6 hours i.e., average number of applications per hour in the last 6 hours. Ranges between [−211, 24763].|Numerical|App|12|
|12|velocity_24h|Velocity of total applications made in last 24 hours i.e., average number of applications per hour in the last 24 hours. Ranges between [1329, 9527].|Numerical|App|1400|
|13|velocity_4w| Velocity of total applications made in last 4 weeks, i.e., average number of applications per hour in the last 4 weeks. Ranges between [2779, 7043].|Numerical|App|2779|
|14|bank_branch_count_8w| Number of total applications in the selected bank branch in last 8 weeks. Ranges between [0, 2521].|Numerical|App|12|
|15|date_of_birth_distinct_emails_4w|Number of emails for applicants with same date of birth in last 4 weeks. Ranges between [0, 42].|Numerical|Emails|12|
|16|employment_status|Employment status of the applicant. 7 possible (annonymized) values.|Categorical|N/A|CA|
|17|credit_risk_score|Internal score of application risk. Ranges between [−176, 387].|Numerical|N/A|-100|
|18|email_is_free|Domain of application email (either free or paid).|Numerical|N/A|1|
|19|housing_status|Current residential status for applicant. 7 possible (annonymized) values.|Categorical|N/A|BC|
|20|phone_home_valid|Validity of provided home phone.|Numerical|N/A|1|
|21|phone_mobile_valid|Validity of provided mobile phone.|Numerical|N/A|1|
|22|bank_months_count|How old is previous account (if held) in months. Ranges between [−1, 31] months (-1 is a missing value).|Numerical|Month|1|
|23|has_other_cards|If applicant has other cards from the same banking company.|Numerical|N/A|1|
|24|proposed_credit_limit|Applicant’s proposed credit limit. Ranges between [200, 2000].|Numerical|USD|200|
|25|foreign_request|If origin country of request is different from bank’s country.|Numerical|N/A||
|26|source|Online source of application. Either browser(INTERNET) or mobile app (APP).|Categorical|N/A|Internet|
|27|session_length_in_minutes|Length of user session in banking website in minutes. Ranges between [−1, 107] minutes|Numerical|Minutes|12|
|28|device_os|Operative system of device that made request. Possible values are: Windows, Macintox, Linux, X11, or other.|Categorical|N/A|Windows|
|29|keep_alive_session|User option on session logout.|Numerical|N/A|1|
|30|device_distinct_emails_8w|Number of distinct emails in banking website from the used device in last 8 weeks. Ranges between [0, 3].|Numerical|Emails|2|
|31|device_fraud_count|Number of fraudulent applications with used device. Ranges between [0, 1].|Numerical|N/A|0|
|32|month|Month where the application was made. Ranges between [0, 7].|Numerical|Month|2|
