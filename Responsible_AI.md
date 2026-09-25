# Responsible AI Report
## PVR Technologies - Movie Success Rate Predictor

### 1. Purpose

The system predicts the potential success category of a movie
using historical movie-related data and machine learning.

### 2. Fairness

The model may reflect biases present in historical movie data.
Special care should be taken with information related to actors,
directors, production companies and geographical regions.

### 3. Privacy

The project uses movie-related public/organizational data.
Personally sensitive information should not be collected or exposed.

### 4. Consent

Only appropriately sourced and permitted datasets should be used.
Data sources and licensing should be documented.

### 5. Explainability

SHAP is used to explain which features contribute to predictions.

### 6. Data Quality

Missing values, inconsistent records and unusual values should
be checked before model training and prediction.

### 7. Bias

Historical success patterns may contain representation and
selection biases. Model results should therefore not be treated
as guaranteed outcomes.

### 8. Human Oversight

Predictions are decision-support outputs and should be reviewed
by humans rather than being treated as certain outcomes.

### 9. Limitations

Movie success can depend on factors such as audience preferences,
marketing, competition, distribution and unexpected events.
These factors may not be fully represented in the dataset.

### 10. Monitoring

Model performance and input-data drift should be monitored after
deployment.