# Credit Score Prediction using Feedforward Neural Network

## 📋 Project Overview

This project implements a **Feedforward Neural Network (FNN)** to predict customer credit score categories using financial and personal data. The model classifies customers into credit score categories (Poor, Standard, Good) and can also predict the probability of default.

## 🎯 Objective

Predict a customer's credit score category based on their financial behavior, spending patterns, and demographic information to assist in credit risk assessment and lending decisions.

## 📊 Dataset Features

The dataset includes comprehensive customer financial data:

- **Financial Metrics**: Income, Savings, Debt, and various ratios
- **Spending Categories**: Clothing, Education, Entertainment, Groceries, Health, Housing, Utilities, Travel, etc.
- **Spending Patterns**: 12-month and 6-month transaction totals with ratios to income, savings, and debt
- **Categorical Features**: Gambling behavior, Debt level, Credit card ownership, Mortgage status, Savings account, Dependents
- **Target Variables**: 
  - Credit Score (Multi-class: Poor, Standard, Good)
  - Default (Binary: 0 or 1)

## 🏗️ Model Architecture

### Feedforward Neural Network Structure

**Input Layer**
- Takes all customer features (numeric + encoded categorical features)
- Feature scaling using StandardScaler for normalized inputs

**Hidden Layers**
- Layer 1: 128 neurons with ReLU activation
- Layer 2: 64 neurons with ReLU activation  
- Layer 3: 32 neurons with ReLU activation
- Each layer includes:
  - Dropout (30%) for regularization
  - Batch Normalization for stable training

**Output Layer**
- Multi-class: Softmax activation for credit score categories
- Binary: Sigmoid activation for default prediction

**Optimization**
- Optimizer: Adam
- Loss Function: 
  - Categorical Cross-Entropy (multi-class)
  - Binary Cross-Entropy (binary classification)
- Callbacks: Early Stopping, Learning Rate Reduction

## 🔧 Implementation Details

### Data Preprocessing
1. **Feature Engineering**: Separation of numerical and categorical features
2. **Encoding**: Label encoding for categorical variables
3. **Scaling**: StandardScaler normalization for all features
4. **Splitting**: 80-20 train-test split with stratification
5. **One-hot Encoding**: Target variables converted for neural network compatibility

### Training Process
- Batch Size: 32
- Maximum Epochs: 100
- Validation Split: 20%
- Early Stopping: Patience of 15 epochs
- Learning Rate Reduction: Factor of 0.5 with patience of 5 epochs

## 📈 Evaluation Metrics

The model is evaluated using comprehensive metrics:

### Multi-class Classification (Credit Score)
- **Accuracy**: Overall classification accuracy
- **Precision**: Correctness of positive predictions per class
- **Recall**: Ability to find all positive instances per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs actual

### Binary Classification (Default Prediction)
- **Accuracy**: Overall prediction accuracy
- **Precision & Recall**: For both default and non-default cases
- **Confusion Matrix**: True/False positives and negatives
- **Probability Scores**: Default probability (0-1 scale)

## 📊 Visualizations

The notebook includes comprehensive visualizations:
- Credit score distribution analysis
- Default rate distribution
- Training & validation accuracy curves
- Training & validation loss curves
- Confusion matrices (raw counts and normalized percentages)
- Per-class performance metrics comparison
- Model architecture diagram

## 🚀 How to Use

### Prerequisites
```
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Running the Notebook
1. Upload the notebook to Google Colab or Jupyter environment
2. Upload your dataset CSV file
3. Run all cells sequentially
4. Models will be saved as `.h5` files
5. Scalers and encoders saved as `.pkl` files

### Making Predictions
The notebook includes a prediction function that accepts customer features and returns:
- Predicted credit score category
- Probability distribution across all categories
- Default probability (if using binary model)

## 💾 Model Outputs

After training, the following files are saved:
- `credit_score_model.h5` - Multi-class classification model
- `default_prediction_model.h5` - Binary classification model  
- `scaler.pkl` - Fitted StandardScaler
- `label_encoder.pkl` - Label encoder for target classes

## 🎯 Use Cases

This model can be applied to:
- **Credit Risk Assessment**: Evaluate creditworthiness of loan applicants
- **Lending Decisions**: Support automated loan approval processes
- **Portfolio Management**: Segment customers by credit risk
- **Financial Planning**: Identify customers needing financial counseling
- **Fraud Detection**: Flag unusual spending patterns
- **Marketing**: Target customers for appropriate financial products

## 📝 Key Features

✅ **Comprehensive Data Preprocessing** - Handles both numerical and categorical features  
✅ **Dual Model Approach** - Multi-class credit scoring + binary default prediction  
✅ **Regularization Techniques** - Dropout and batch normalization prevent overfitting  
✅ **Advanced Callbacks** - Early stopping and learning rate scheduling  
✅ **Extensive Evaluation** - Multiple metrics and visualizations  
✅ **Production Ready** - Model saving and prediction pipeline included  
✅ **Well Documented** - Clear explanations and comments throughout

## 🔍 Model Interpretation

The feedforward neural network learns complex, non-linear relationships between:
- Income and spending ratios
- Debt management patterns
- Category-wise expenditure behavior
- Financial responsibility indicators
- Demographic factors

These patterns are captured through multiple hidden layers that progressively extract higher-level features from the raw data.

## 🎓 Learning Outcomes

This project demonstrates:
- Building neural networks with TensorFlow/Keras
- Handling imbalanced classification problems
- Feature engineering for financial data
- Model evaluation and validation techniques
- Hyperparameter tuning and optimization
- Production deployment preparation

## 📌 Future Enhancements

Potential improvements:
- Hyperparameter tuning with Grid/Random Search
- SMOTE for handling class imbalance
- Feature importance analysis using SHAP values
- Ensemble methods (combine with XGBoost, Random Forest)
- Cross-validation for robust performance estimates
- API deployment using Flask/FastAPI
- Real-time prediction dashboard

## 📄 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

Created as a demonstration of deep learning for financial risk assessment.

---

**Note**: This model is for educational and demonstration purposes. For production use in financial institutions, additional validation, regulatory compliance, and risk assessment procedures are required.
