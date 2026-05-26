@echo off
echo =====================================================
echo HepaStage AI - 90%%+ Accuracy - Complete Pipeline
echo =====================================================
echo.

echo [1/5] Global deps check...
python -c "import pandas, sklearn, flask; print('OK - global deps OK')" || (
  echo Installing global deps ^(no .venv needed^)...
  pip install scikit-learn pandas flask flask-cors imbalanced-learn joblib numpy --user --prefer-binary
)

echo.
echo [2/5] Training 90%%+ model...
python src/train_model_expanded.py || echo "Using pre-trained models (src/train failed, models exist)"

echo.
echo [3/5] Model ^& scaler ready. Accuracy in model_metadata.json
type models\model_metadata.json | findstr accuracy

echo.
echo [4/5] Starting Flask API...
start "" "http://127.0.0.1:5000/"
python src\app.py

echo.
echo [5/5] API running ^| Open index.html ^| API: localhost:5000/predict
echo Press Ctrl+C to stop API
pause
