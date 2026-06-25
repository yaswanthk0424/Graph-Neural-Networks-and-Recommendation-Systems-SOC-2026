# Task 3: Written Analysis

## Deconstructing the Evaluation Metrics

The performance of each classifier was evaluated using Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

* **Accuracy** represents the proportion of correctly classified submissions among all submissions. For this problem, it indicates the overall percentage of submissions that were correctly identified as either late or on time.

* **Precision (Class 1)** measures the proportion of submissions predicted as **Late** that were actually late. A high precision means that when the model predicts a late submission, it is usually correct, reducing false alarms.

* **Recall (Class 1)** measures the proportion of actual late submissions that were correctly identified by the model. In this context, a **low recall** means that many genuinely late submissions are incorrectly classified as on-time. As a result, the institution may fail to identify students who require timely reminders or intervention.

* **F1-Score (Class 1)** is the harmonic mean of Precision and Recall. It provides a balanced measure of model performance, particularly when both false positives and false negatives are important.

* **AUC-ROC** measures the model's ability to distinguish between late and on-time submissions across different decision thresholds. A value closer to 1 indicates excellent discrimination, while a value close to 0.5 indicates performance similar to random guessing.

---

## Model Selection

Among the evaluated models, **Random Forest** achieved the best overall performance based on the evaluation metrics.

This model performed better because it was able to capture the relationships between the selected features more effectively than the other models. While Logistic Regression assumes a linear decision boundary and Support Vector Machines require careful tuning of hyperparameters, **Random Forest** can naturally learn complex non-linear interactions between features and is generally robust to noise and overfitting due to its ensemble of decision trees.


---

## Practical Application

This predictive model can assist educational institutions or organizations in identifying submissions that are likely to be delayed before the actual deadline. Early identification enables instructors or administrators to send reminders, provide additional academic support, allocate resources more efficiently, and monitor students who may be at risk of consistently submitting assignments late. Such proactive intervention can improve submission rates and overall course management.

---

## Effect of Using More Features

An additional experiment was performed by training the models using all available features instead of only the top 15 selected features.

Using all features may improve predictive performance if the additional features contain useful information. However, it can also introduce redundant or noisy features, increasing computational cost and potentially reducing model interpretability. Using only the top 15 correlated features provides a simpler model with lower training time while often achieving performance comparable to using the complete feature set.

The comparison between both approaches helps determine whether feature selection improves efficiency without sacrificing predictive accuracy.
