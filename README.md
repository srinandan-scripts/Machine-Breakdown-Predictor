# Breakdown Prediction System

## Features
- Predict machine breakdown likelihood using ensemble models
- Handles multiple years of data automatically
- Shift-aware, holiday-aware predictions
- Weekly forecast window (configurable)
- Outputs prediction accuracy and feature importance

## Setup
- Edit `config.yaml` to adjust holidays, shift definitions, etc.
- Run: `Predict Breakdown Master.py`

## Requirements
- pandas, numpy, xgboost, scikit-learn, matplotlib, yaml, holidays
