# 🌤️ Guwahati Weather Prediction System

An AI-powered weather forecasting application using 50 years of historical data (1973-2023) from Guwahati, India.

## 🚀 Features

- **📈 Historical Analysis**: Explore 50+ years of weather data with interactive visualizations
- **🔮 Future Predictions**: AI-powered 30-day weather forecasts
- **🤖 Model Performance**: Compare 6 different ML models (XGBoost, Random Forest, LSTM, etc.)
- **🎯 Custom Predictions**: Input your own parameters for instant predictions
- **📊 Climate Insights**: Analyze climate change trends and extreme weather events

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: XGBoost, Random Forest, LSTM Neural Networks
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deep Learning**: TensorFlow/Keras

## 📦 Installation

### Local Setup

1. **Clone or download this repository**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Ensure all required files are present:**

   - `app.py` - Main application file
   - `xgboost_weather_model.pkl` - XGBoost model
   - `random_forest_weather_model.pkl` - Random Forest model
   - `lstm_weather_model.h5` - LSTM model
   - `feature_scaler.pkl` - Feature scaler
   - `feature_columns.pkl` - Feature column names
   - `future_predictions.csv` - Pre-generated predictions
   - `model_comparison_results.csv` - Model performance metrics
   - All CSV files: `guwahati YYYY-01-01 to YYYY-12-31.csv` (1973-2023)

4. **Run the application**

```bash
streamlit run app.py
```

5. **Open browser** at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

1. **Push to GitHub**

   - Create a new GitHub repository
   - Push all files including models and data

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

## 📊 Model Performance

| Model             | RMSE (°C) | MAE (°C)  | R² Score  |
| ----------------- | --------- | --------- | --------- |
| XGBoost           | Best      | Best      | Best      |
| Random Forest     | Excellent | Excellent | Excellent |
| LSTM              | Very Good | Very Good | Very Good |
| Gradient Boosting | Good      | Good      | Good      |
| Ridge Regression  | Baseline  | Baseline  | Baseline  |
| Linear Regression | Baseline  | Baseline  | Baseline  |

## 🎯 Usage

### 🏠 Home Page

- View key weather statistics
- See recent temperature trends
- Quick climate change indicators

### 📈 Historical Analysis

- Select date ranges
- Analyze temperature trends over decades
- Explore monthly patterns

### 🔮 Future Predictions

- View 30-day weather forecasts
- Compare with historical data
- Download predictions as CSV

### 🤖 Model Performance

- Compare different ML models
- View performance metrics
- Understand model explanations

### 🎯 Custom Prediction

- Input custom weather parameters
- Get instant predictions
- See confidence intervals

### 📊 Insights

- Climate change analysis
- Seasonal patterns
- Extreme weather events
- Key findings and trends

## 📁 Project Structure

```
Guwahati_weather_1973-2023/
├── app.py                              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── xgboost_weather_model.pkl          # Trained XGBoost model
├── random_forest_weather_model.pkl    # Trained Random Forest model
├── lstm_weather_model.h5              # Trained LSTM model
├── feature_scaler.pkl                 # Feature scaling object
├── feature_columns.pkl                # Feature column names
├── future_predictions.csv             # Pre-generated predictions
├── model_comparison_results.csv       # Model performance results
└── guwahati *.csv                     # Historical data files (51 files)
```

## 🔧 Troubleshooting

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Memory Issues

If you encounter memory issues with large datasets, consider:

- Reducing the number of historical years loaded
- Using data sampling for visualizations
- Increasing system memory allocation

### Model Loading Errors

Ensure all `.pkl` and `.h5` files are in the same directory as `app.py`

## 📈 Future Enhancements

- [ ] Real-time weather API integration
- [ ] Hourly predictions
- [ ] Weather alerts and notifications
- [ ] Mobile-responsive design improvements
- [ ] User authentication
- [ ] Save favorite locations
- [ ] Export reports as PDF
- [ ] Multi-city support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ for weather prediction enthusiasts

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

### 🌟 Star this repository if you find it helpful!
