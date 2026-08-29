BMDS2003 Data Science - Telco Customer Churn Prediction
=======================================================

CONTENTS
  BMDS2003_Telco_Churn_Modelling_v3.ipynb  full CRISP-DM notebook (Parts 1-6)
  Telco_Cusomer_Churn.csv                  raw dataset
  telco_churn_cleaned.csv                  cleaned dataset
  telco_churn_model_ready.csv              encoded model-ready matrix
  model_comparison_results.csv             evaluation metrics for all 4 models
  trained_models.pkl                       4 trained models + scaler + metadata
  requirements.txt                         pinned dependencies (see NOTE below)

  DEPLOYMENT PROTOTYPE (Streamlit)
  app.py                                   layout, widgets and page structure
  analysis.py                              data loading and model computations
  charts.py                                interactive Plotly figures
  theme.py                                 shared palette
  .streamlit/config.toml                   theme - single source of truth for colour

BEST MODEL: Random Forest

THE CHARTS ARE GENERATED LIVE
  The prototype no longer ships the notebook's static PNG figures. Every chart
  is rebuilt at runtime from the CSVs and the trained models, so the reader can
  hover for exact values, zoom, toggle series and re-run the evaluation at a
  different decision threshold.

  The app rebuilds the notebook's exact train/test split
  (test_size=0.2, random_state=42, stratify=y). This was verified by
  recomputing all four models' test metrics and matching
  model_comparison_results.csv to four decimal places, so the numbers shown in
  the app agree with the numbers in the report.

TO RUN THE PROTOTYPE
  1. Keep app.py, analysis.py, charts.py, theme.py, .streamlit/config.toml,
     trained_models.pkl, telco_churn_cleaned.csv and
     telco_churn_model_ready.csv in the SAME directory.
  2. pip install -r requirements.txt
  3. streamlit run app.py

NOTE ON DEPENDENCIES - PLEASE DO NOT UNPIN
  requirements.txt is pinned deliberately. Three of the pins are load-bearing:
    scikit-learn 1.6.1  the exact version that trained trained_models.pkl
    xgboost      3.4.1  version 3.3.0 CANNOT read the saved booster - it fails
                        with XGBoostError "input stream corrupted"
    numpy 2.x / pandas 2.x
                        the pickle stores numpy._core arrays and pandas 2.x
                        DataFrame internals; numpy 1.x and pandas 3.x cannot
                        load it
  Use Python 3.12 (set this in Streamlit Cloud's Advanced settings). Note that
  scikit-learn 1.6.1 has no wheels for Python 3.14.
