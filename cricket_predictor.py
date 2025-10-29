import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class CricketScorePredictor:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
    
    def train_models(self, X_train, y_train, feature_names):
        """Train multiple models and compare performance"""
        print("Training models...")
        self.feature_names = feature_names
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"{name} trained successfully!")
            except Exception as e:
                print(f"Error training {name}: {e}")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        results = {}
        
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'predictions': y_pred
                }
                
                print(f"\n{name.upper()} Results:")
                print(f"MAE: {mae:.2f}")
                print(f"RMSE: {rmse:.2f}")
                print(f"R² Score: {r2:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Select best model based on lowest RMSE
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
            self.best_model = self.trained_models[best_model_name]
            self.best_model_name = best_model_name
            print(f"\nBest model: {best_model_name}")
        
        return results
    
    def predict_score(self, match_data):
        """Predict score for new match data"""
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        prediction = self.best_model.predict(match_data)
        return prediction
    
    def plot_predictions(self, y_test, results, save_path='predictions_comparison.png'):
        """Plot actual vs predicted scores for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(y_test, result['predictions'], alpha=0.6)
            
            # Perfect prediction line
            min_val = min(min(y_test), min(result['predictions']))
            max_val = max(max(y_test), max(result['predictions']))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f'{name.title()}\nR² = {result["R2"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Predictions plot saved as {save_path}")
    
    def plot_feature_importance(self, save_path='feature_importance.png'):
        """Plot feature importance for tree-based models"""
        if self.best_model_name in ['random_forest', 'xgboost']:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                
                # Create feature importance dataframe
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Plot
                plt.figure(figsize=(10, 8))
                sns.barplot(data=feature_imp.head(15), x='importance', y='feature')
                plt.title(f'Top 15 Feature Importances - {self.best_model_name.title()}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Feature importance plot saved as {save_path}")
                
                return feature_imp
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath='best_cricket_model.pkl'):
        """Save the best trained model"""
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            print(f"Best model ({self.best_model_name}) saved as {filepath}")
        else:
            print("No trained model to save")
    
    def load_model(self, filepath='best_cricket_model.pkl'):
        """Load a saved model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            print(f"Model ({self.best_model_name}) loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_live_score(self, current_overs, current_runs, current_wickets, 
                          venue='Unknown', batting_team='Unknown', bowling_team='Unknown'):
        """Predict final score based on current match situation"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Create feature vector (simplified version)
        features = np.array([[
            current_overs,
            current_runs,
            current_wickets,
            current_runs / (current_overs + 0.1),  # current run rate
            (20 - current_overs) * 6,  # balls remaining
            10 - current_wickets,  # wickets remaining
            1 if current_overs <= 6 else 0,  # powerplay
            1 if current_overs >= 16 else 0,  # death overs
            1 if 6 < current_overs < 16 else 0,  # middle overs
        ]])
        
        # Pad with zeros if more features are expected
        if len(self.feature_names) > features.shape[1]:
            padding = np.zeros((1, len(self.feature_names) - features.shape[1]))
            features = np.hstack([features, padding])
        
        prediction = self.best_model.predict(features)[0]
        return max(current_runs, prediction)  # Ensure prediction is at least current score