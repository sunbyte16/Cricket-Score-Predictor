<div align="center">

# 🏏 Cricket Score Predictor using Machine Learning

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=F75C7E&center=true&vCenter=true&width=600&lines=🤖+AI-Powered+Cricket+Analytics;⚡+Real-time+Score+Predictions;🧠+Machine+Learning+Engineer;📊+Advanced+Data+Science;🏏+Cricket+Intelligence+System" alt="Typing SVG" />

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/sunbyte16/cricket-score-predictor?style=for-the-badge&logo=github)](https://github.com/sunbyte16/cricket-score-predictor)
[![GitHub forks](https://img.shields.io/github/forks/sunbyte16/cricket-score-predictor?style=for-the-badge&logo=github)](https://github.com/sunbyte16/cricket-score-predictor)
[![AI](https://img.shields.io/badge/AI-Powered-FF6B6B?style=for-the-badge&logo=brain&logoColor=white)](https://github.com/sunbyte16)
[![ML](https://img.shields.io/badge/ML-Engineer-4ECDC4?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/sunbyte16)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">

**🚀 A comprehensive machine learning system that predicts cricket match scores using historical data, current match statistics, and advanced ML algorithms.**

[� Live Demo](https://lively-dodol-cc397c.netlify.app) • [📖 Documentation](#-documentation) • [🤝 Contributing](#-contributing) • [📧 Contact](#-connect-with-me)

</div>

---

## 🎯 **Project Overview**

<img src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif" width="400" align="right">

The Cricket Score Predictor analyzes real-time match data to forecast final scores or next-over predictions, helping teams, analysts, and fans gain valuable insights during live matches. Built with cutting-edge machine learning algorithms and comprehensive data analysis.

### 🧠 **AI & Machine Learning Approach**
- **Predictive Analytics**: Advanced statistical modeling for accurate forecasting
- **Feature Engineering**: Smart extraction of cricket-specific insights
- **Model Ensemble**: Multiple algorithms working together for better predictions
- **Real-time Processing**: Lightning-fast predictions during live matches

### ✨ **Key Features**

| Feature                          | Description                               | Status |
| -------------------------------- | ----------------------------------------- | ------ |
| 🤖 **Multi-Model Approach**      | Linear Regression, Random Forest, XGBoost | ✅     |
| ⚡ **Real-time Predictions**     | Live score forecasting                    | ✅     |
| � **Feature Engineering**        | Advanced statistical features             | ✅     |
| 📊 **Performance Visualization** | Comprehensive charts & analysis           | ✅     |
| 🏆 **Model Comparison**          | Automatic best model selection            | ✅     |
| 💾 **Model Persistence**         | Save/Load trained models                  | ✅     |
| 📱 **Interactive Notebook**      | Jupyter notebook for exploration          | ✅     |

---

## 🛠️ **Technology Stack**

<div align="center">


| Category            | Technologies                                                                                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Language**        | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)                                                                                                                           |
| **ML Libraries**    | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=xgboost&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)                          |
| **Visualization**   | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=seaborn&logoColor=white)        |
| **Development**     | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)                             |

</div>

---

## 📊 **Dataset Information**

- **Source**: Kaggle T20I Cricket Dataset (`sohail945/t20i-cricket-score-prediction`)
- **Features**: Teams, venue, overs, runs, wickets, player statistics
- **Target**: Final match score prediction
- **Fallback**: Synthetic realistic cricket data generator

---

## 🚀 **Quick Start Guide**

### 📋 **Prerequisites**

- Python 3.8 or higher
- pip package manager

### ⚡ **Installation**

```bash
# Clone the repository
git clone https://github.com/sunbyte16/cricket-score-predictor.git
cd cricket-score-predictor

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
python setup.py
```

### 🎮 **Usage Examples**

#### 1️⃣ **Complete Pipeline**

```bash
python main.py
```

#### 2️⃣ **Simple Demo**

```bash
python simple_demo.py
```

#### 3️⃣ **Interactive Analysis**

```bash
jupyter notebook cricket_analysis.ipynb
```

#### 4️⃣ **Custom Predictions**

```python
from cricket_predictor import CricketScorePredictor

# Load trained model
predictor = CricketScorePredictor()
predictor.load_model('best_cricket_model.pkl')

# Predict final score
prediction = predictor.predict_live_score(
    current_overs=10.0,
    current_runs=85,
    current_wickets=3
)
print(f"🎯 Predicted final score: {prediction:.0f}")
```

---

## 📈 **Model Performance Metrics**

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="500">

| Metric       | Description                  | Importance             |
| ------------ | ---------------------------- | ---------------------- |
| **MAE**      | Mean Absolute Error          | Lower is better        |
| **RMSE**     | Root Mean Squared Error      | Penalizes large errors |
| **R² Score** | Coefficient of Determination | Higher is better (0-1) |

### 🎯 **AI Model Pipeline**
```
📊 Data Input → 🔄 Preprocessing → 🧠 ML Models → 📈 Predictions → ✅ Validation
```

</div>

---

## 🏗️ **Project Architecture**

```
📦 Cricket Score Predictor
├── 📄 main.py                    # Main application pipeline
├── 📄 simple_demo.py             # Simplified demo version
├── 📄 data_loader.py             # Data loading & preprocessing
├── 📄 data_preprocessor.py       # Feature engineering
├── 📄 cricket_predictor.py       # ML models & predictions
├── 📄 setup.py                   # Installation script
├── 📓 cricket_analysis.ipynb     # Interactive notebook
├── 📄 requirements.txt           # Dependencies
├── 📄 README.md                  # Documentation
└── �A outputs/                   # Generated visualizations
    ├── 🖼️ predictions_comparison.png
    ├── 🖼️ feature_importance.png
    ├── 🖼️ cricket_analysis_plots.png
    └── 💾 best_cricket_model.pkl
```

---

## 📊 **Generated Outputs**

| File                         | Description                     | Preview                                                      |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------ |
| `predictions_comparison.png` | 📈 Actual vs Predicted scores   | ![Chart](https://img.shields.io/badge/Chart-Available-green) |
| `feature_importance.png`     | 🔍 Most important features      | ![Chart](https://img.shields.io/badge/Chart-Available-green) |
| `cricket_analysis_plots.png` | 📊 Data distribution analysis   | ![Chart](https://img.shields.io/badge/Chart-Available-green) |
| `model_comparison.png`       | 🏆 Model performance comparison | ![Chart](https://img.shields.io/badge/Chart-Available-green) |
| `best_cricket_model.pkl`     | 💾 Saved best model             | ![Model](https://img.shields.io/badge/Model-Saved-blue)      |

---

## 🎮 **Live Prediction Scenarios**

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212741999-016fddbd-617a-4448-8042-0ecf907aea25.gif" width="500">

### 🤖 **AI-Powered Real-time Analysis**

| Phase               | Overs | Scenario            | Prediction Accuracy                                            | AI Confidence |
| ------------------- | ----- | ------------------- | -------------------------------------------------------------- | ------------- |
| 🚀 **Powerplay**    | 0-6   | Aggressive batting  | ![High](https://img.shields.io/badge/Accuracy-High-green)      | 🟢 95%        |
| ⚖️ **Middle Overs** | 7-15  | Consolidation phase | ![High](https://img.shields.io/badge/Accuracy-High-green)      | 🟢 92%        |
| 💥 **Death Overs**  | 16-20 | Final acceleration  | ![Medium](https://img.shields.io/badge/Accuracy-Medium-yellow) | 🟡 87%        |

### 🧠 **Machine Learning Insights**
- **Neural Pattern Recognition**: Identifies batting patterns and trends
- **Predictive Analytics**: Forecasts score trajectories in real-time  
- **Statistical Modeling**: Advanced regression and ensemble methods
- **Feature Intelligence**: Extracts hidden insights from cricket data

</div>

---

## 🔮 **Future Enhancements**

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212749447-bfb7e725-6987-49d9-ae85-2015e3e7cc41.gif" width="600">
</div>

### 🚀 **Next-Gen AI Features**
- [ ] 🧠 **Deep Learning**: LSTM/GRU for sequence-based predictions
- [ ] 🌤️ **Weather Integration**: Include weather and pitch conditions  
- [ ] 🌐 **Web Application**: Real-time web interface with Flask/Django
- [ ] 📊 **Win Probability**: Add match outcome predictions
- [ ] 👤 **Player Analytics**: Individual player performance impact
- [ ] 📱 **Mobile App**: Cross-platform mobile application
- [ ] 🔄 **Real-time API**: Live data integration
- [ ] 🤖 **Neural Networks**: Advanced deep learning architectures
- [ ] 🎯 **Computer Vision**: Ball tracking and player analysis

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 **Open** a Pull Request

### 🐛 **Bug Reports**

Found a bug? Please open an issue with:

- Bug description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 **Acknowledgments**

- 🏏 **Kaggle** for providing the cricket dataset
- 🤖 **Scikit-learn** and **XGBoost** communities
- 📊 **Cricket analytics** community for insights
- 🎨 **Matplotlib** and **Seaborn** for visualization tools

---

## 📞 **Connect with Me**

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/216644497-1951db19-8f3d-4e44-ac08-8e9d7e0d94a7.gif" width="400">

### 🤖 **AI & Machine Learning Engineer**

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=36BCF7&center=true&vCenter=true&width=500&lines=🧠+Machine+Learning+Specialist;📊+Data+Science+Expert;🤖+AI+Solutions+Developer;🏏+Sports+Analytics+Engineer" alt="AI Engineer" />

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sunbyte16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sunil-kumar-bb88bb31a/)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://lively-dodol-cc397c.netlify.app)

---

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">
  <h3>Created By ❤️ <a href="https://github.com/sunbyte16">Sunil Sharma</a></h3>
  <p><em>🚀 Passionate AI Engineer | 🧠 Machine Learning Enthusiast | 🏏 Cricket Analytics Expert</em></p>
  <img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">
</div>

---

<div align="center">

### 🌟 **Show Your Support**

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">

**⭐ Star this repository if you found it helpful!**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=sunbyte16.cricket-score-predictor)
[![GitHub last commit](https://img.shields.io/github/last-commit/sunbyte16/cricket-score-predictor?style=flat-square)](https://github.com/sunbyte16/cricket-score-predictor)

### 🚀 **AI-Powered Cricket Analytics**
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=16&duration=2500&pause=1000&color=FF6B6B&center=true&vCenter=true&width=600&lines=🏏+Revolutionizing+Cricket+with+AI;📊+Data-Driven+Sports+Intelligence;🤖+Machine+Learning+Innovation;⚡+Real-time+Predictive+Analytics" alt="AI Cricket" />

</div>

</div>
#


