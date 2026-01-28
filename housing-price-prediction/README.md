# Housing Price Prediction using Linear Regression Models

A machine learning project that predicts housing prices based on various property features using Linear Regression, Ridge Regression, and Lasso Regression models. The project includes an interactive Gradio web interface for real-time price predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project demonstrates the application of regression techniques to predict house prices. It uses synthetic data to train multiple linear regression models and provides an interactive interface for making predictions based on house characteristics.

## 📊 Dataset

The dataset contains **55,000 synthetic housing records** with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| `house_size_sqft` | Size of the house in square feet | 1,000 - 5,000 sqft |
| `bedrooms` | Number of bedrooms | 1 - 6 |
| `age_years` | Age of the house in years | 0 - 50 years |
| `distance_to_center_km` | Distance from city center | 1 - 30 km |
| `bathrooms` | Number of bathrooms | 1.0 - 4.5 |
| `price_usd` | House price (target variable) | $78K - $1M |

### Data Generation Formula
```
Price = 50,000 + 150×size + 20,000×bedrooms - 800×age - 2,000×distance + 15,000×bathrooms + noise
```

## ✨ Features

- **Data Exploration**: Comprehensive EDA with visualizations (histograms, box plots)
- **Multiple Models**: Linear Regression, Ridge Regression, Lasso Regression
- **Model Comparison**: Performance evaluation using R² score and MSE
- **Interactive Interface**: Gradio-based web UI for real-time predictions
- **Train-Test Split**: 80-20 split for robust model evaluation

## 🤖 Models Implemented

### 1. Linear Regression
- **R² Score**: 0.9721
- **MSE**: 906,740,243.57
- Base model without regularization

### 2. Lasso Regression (L1 Regularization)
- **Alpha**: 1.4
- **R² Score**: 0.9721
- Helps with feature selection

### 3. Ridge Regression (L2 Regularization)
- **Alpha**: 2.5
- **R² Score**: 0.9721
- Prevents overfitting through coefficient shrinkage

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. **Install required packages**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn gradio
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**
```bash
jupyter notebook linear_regression_project.ipynb
```

2. **Run all cells sequentially** to:
   - Generate the dataset
   - Train the models
   - Launch the Gradio interface

### Using the Gradio Interface

Once the last cell is executed, a local web interface will launch where you can:

1. Adjust sliders for each feature:
   - House Size (1000-5000 sqft)
   - Bedrooms (1-6)
   - Age (0-50 years)
   - Distance to Center (1-30 km)
   - Bathrooms (1-4.5)

2. Click **Submit** to get the predicted price

3. Example prediction:
   ```
   House Size: 3000 sqft
   Bedrooms: 4
   Age: 15 years
   Distance: 10 km
   Bathrooms: 3
   
   → Predicted Price: $612,450.32
   ```

## 📈 Results

### Model Performance Comparison

| Model | R² Score | MSE | Best For |
|-------|----------|-----|----------|
| Linear Regression | 0.9721 | 906,740,244 | Baseline predictions |
| Lasso (α=1.4) | 0.9721 | ~906,740,244 | Feature selection |
| Ridge (α=2.5) | 0.9721 | ~906,740,244 | Preventing overfitting |

All three models show excellent performance with R² scores above 97%, indicating strong predictive capability.

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models and metrics
- **Gradio**: Interactive web interface
- **Jupyter Notebook**: Development environment

## 📁 Project Structure

```
housing-price-prediction/
│
├── linear_regression_project.ipynb    # Main notebook
├── housing_linear_regression_data.csv  # Generated dataset
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
└── images/                            # Visualization outputs (optional)
```

## 🔮 Future Improvements

- [ ] Add more complex features (e.g., neighborhood quality, school ratings)
- [ ] Implement advanced models (Random Forest, XGBoost, Neural Networks)
- [ ] Add cross-validation for more robust evaluation
- [ ] Include feature importance analysis
- [ ] Deploy the Gradio app to Hugging Face Spaces
- [ ] Add prediction confidence intervals
- [ ] Implement model persistence (save/load trained models)
- [ ] Create visualizations for predicted vs actual prices

## 📝 Key Insights

1. **Strong Linear Relationship**: The high R² score (0.97) indicates that house prices have a strong linear relationship with the features
2. **Model Stability**: All three models perform similarly, suggesting the data doesn't require heavy regularization
3. **Feature Impact**: House size and number of bedrooms are positive contributors, while age and distance negatively impact price

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Mohammed Habeeb Mohsin Siddiqui
- GitHub: [@MHMSiddiqui](https://github.com/MHMSiddiqui)
- LinkedIn: [Habeeb Mohsin](https://www.linkedin.com/in/habeeb-mohsin-b29225254/)

## 🙏 Acknowledgments

- Dataset: Synthetically generated for educational purposes
- Inspired by real-world housing price prediction challenges
- Built as a demonstration of regression techniques in machine learning

---

**Note**: This project uses synthetic data for demonstration purposes. For real-world applications, use actual housing market data and consider additional factors like location, market trends, and economic indicators.
