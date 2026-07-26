BMDS2003 Data Science - Telco Customer Churn Prediction
=======================================================
CONTENTS
  BMDS2003_Telco_Churn_Modelling_v3.ipynb  full CRISP-DM notebook (Parts 1-6)
  Telco_Cusomer_Churn.csv                  raw dataset
  telco_churn_cleaned.csv                  cleaned dataset
  telco_churn_model_ready.csv              encoded model-ready matrix
  model_comparison_results.csv             evaluation metrics for all 4 models
  trained_models.pkl                       4 trained models + scaler + metadata
  streamlit_app.py                         deployment prototype
  figures/                                 all charts used in the report
  requirements.txt

BEST MODEL: Random Forest

TO RUN THE PROTOTYPE
  1. Keep streamlit_app.py, trained_models.pkl and telco_churn_cleaned.csv
     in the SAME directory.
  2. pip install -r requirements.txt
  3. streamlit run streamlit_app.py
