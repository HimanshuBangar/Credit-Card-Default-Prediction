# Credit Risk Intelligence — Credit Card Default Prediction

A Streamlit machine-learning app upgraded from a basic Logistic Regression notebook.

## Features
- Overview dashboard and dataset distribution
- Compare Logistic Regression, Random Forest, and Histogram Gradient Boosting
- Individual customer default-probability estimate
- Batch CSV predictions with downloadable report
- Evaluation dashboard: precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
- Permutation-based feature importance
- Project overview, workflow, limitations, and future scope

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

The app downloads the public YBI Foundation demo dataset when it starts. If the URL is inaccessible, upload a compatible CSV from the sidebar.

## Dataset schema
Training CSV must contain:
- `Income`
- `Age`
- `Loan`
- `Loan to Income`
- `Default` (0 = no default, 1 = default)

For batch predictions, only the four input columns are required. A template can be downloaded from the app.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. In Streamlit Community Cloud, create a new app and select the repository, branch, and `app.py`.
3. Deploy.

## Important notes
- Model metrics are computed at runtime and can change if the dataset or split changes.
- The dataset is a small educational sample, not suitable for real credit decisions.
- Risk bands are illustrative, not official financial thresholds.
- The app is a decision-support demonstration and should not be used to approve or reject real credit applications.
