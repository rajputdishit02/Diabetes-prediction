# Diabetes Prediction App

A machine learning web application that predicts diabetes risk using a RandomForest classifier, built with Streamlit.

## Features

- **EDA (Exploratory Data Analysis)**: Visualize dataset statistics, distributions, and correlations
- **Prediction**: Input patient details and get real-time diabetes risk predictions
- **Interactive UI**: User-friendly Streamlit interface with sliders for easy input

## Project Structure

```
.
├── app.py                    # Streamlit web app (loads pre-trained model)
├── main.py                   # Full pipeline (trains model + runs app)
├── diabetes_model.pkl        # Pre-trained RandomForest model
├── data/
│   └── diabetes.csv         # Dataset
└── .gitignore               # Git ignore rules
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Diabetes-prediction.git
   cd Diabetes-prediction
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit pandas seaborn matplotlib scikit-learn
   ```

## Usage

### Run the App (uses pre-trained model):
```bash
streamlit run app.py
```

### Train & Run (retrains model every time):
```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

## Model Details

- **Algorithm**: RandomForest Classifier
- **Train/Test Split**: 80/20
- **Features**: 8 patient health metrics
- **Target**: Diabetes diagnosis (0 = No, 1 = Yes)

## Dataset

- **Source**: diabetes.csv
- **Samples**: 768 patient records
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## Notes

- The model file (`diabetes_model.pkl`) is excluded from Git (see `.gitignore`)
- Data files are also excluded to keep the repository lightweight
- Retrain the model periodically with `main.py` to improve predictions

## Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn

## License

MIT License
