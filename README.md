# Big Mart Sales Prediction App 🛒

A Streamlit web application that predicts item outlet sales using an XGBoost machine learning model. The app provides an intuitive sidebar interface with friendly feature names that are automatically encoded before being sent to the model.

## Features

✨ **User-Friendly Interface**
- Sidebar input controls with friendly names (e.g., "Low Fat", "Tier 1", "Small")
- Real-time prediction with visual feedback
- Input summary table showing all submitted values
- Responsive layout optimized for desktop browsers

## Prerequisites
- Windows (PowerShell recommended)
- Python 3.8+ (you already have 3.12.2 in the project's `.venv`)
- A virtual environment (recommended) — this project uses `./.venv`
- The model file `BMSP.pkl` must be in the project root (same folder as `app.py`)

## Quick run (PowerShell)

1. If you don't have a venv yet (skip if `.venv` exists):

```powershell
py -3 -m venv .venv
```

2. Allow activation for this session (Process scope only) and activate the venv:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies (either from `requirements.txt` if you have one, or install required packages):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # if you created a requirements file
# OR install the main packages used by the app:
python -m pip install streamlit pandas xgboost
```

4. Ensure `BMSP.pkl` is in the same folder as `app.py`.

5. Run the app with Streamlit:

```powershell
streamlit run app.py
```

## Troubleshooting

If you prefer to run the command without activating the venv, you can run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes & Troubleshooting
- If the app shows an error about the model file missing, place `BMSP.pkl` in the project root next to `app.py`.
- If you see errors while unpickling, check that the `xgboost` version installed is compatible with the model. Re-train or re-pickle with the matching version if necessary.
- To exit the Streamlit server, press Ctrl+C in the terminal where it's running.
- To deactivate the venv, run:

```powershell
deactivate
```

## Optional: create a requirements file
After you install the packages you need, pin them with:

```powershell
python -m pip freeze > requirements.txt
```

## Dependencies

All project dependencies are listed in `requirements.txt`:
- `streamlit` — Web app framework
- `pandas` — Data manipulation
- `xgboost` — Machine learning model
- Plus supporting libraries

---

**Questions or Issues?** Open an issue on the [GitHub repository](https://github.com/HMBinara/Big_Mart_Sales_Prediction_App).
