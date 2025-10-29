#!/usr/bin/env python3
"""
Cricket Score Predictor - Main Application
Predicts cricket match scores using machine learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_cricket_data
from data_preprocessor import CricketDataPreprocessor
from cricket_predictor import CricketScorePredictor

def main():
    """Main function to run the cricket score prediction pipeline"""
    print("=" * 60)
    print("CRICKET SCORE PREDICTOR USING MACHINE LEARNING")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n1. LOADING DATA")
    print("-" * 30)
    df = load_cricket_data()
    
    if df is None or df.empty:
        print("Error: No data loaded. Exiting...")
        return
    
    # Step 2: Data Preprocessing
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    preprocessor = CricketDataPreprocessor()
    processed_data, feature_cols = preprocessor.preprocess_data(df)
    
    # Step 3: Prepare Train-Test Split
    print("\n3. PREPARING TRAIN-TEST SPLIT")
    print("-" * 30)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
        processed_data, feature_cols
    )
    
    # Step 4: Train Models
    print("\n4. TRAINING MODELS")
    print("-" * 30)
    predictor = CricketScorePredictor()
    predictor.train_models(X_train, y_train, feature_cols)
    
    # Step 5: Evaluate Models
    print("\n5. EVALUATING MODELS")
    print("-" * 30)
    results = predictor.evaluate_models(X_test, y_test)
    
    # Step 6: Visualizations
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 30)
    
    # Plot predictions comparison
    predictor.plot_predictions(y_test, results)
    
    # Plot feature importance
    feature_importance = predictor.plot_feature_importance()
    
    # Additional visualizations
    create_additional_plots(processed_data, y_test, results)
    
    # Step 7: Save Model
    print("\n7. SAVING BEST MODEL")
    print("-" * 30)
    predictor.save_model()
    
    # Step 8: Live Prediction Demo
    print("\n8. LIVE PREDICTION DEMO")
    print("-" * 30)
    demo_live_predictions(predictor)
    
    print("\n" + "=" * 60)
    print("CRICKET SCORE PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

def create_additional_plots(data, y_test, results):
    """Create additional visualization plots"""
    
    # 1. Score distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(data['total'] if 'total' in data.columns else y_test, bins=30, alpha=0.7, color='skyblue')
    plt.title('Distribution of Total Scores')
    plt.xlabel('Total Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Run rate vs Total score
    plt.subplot(1, 3, 2)
    if 'current_run_rate' in data.columns and 'total' in data.columns:
        plt.scatter(data['current_run_rate'], data['total'], alpha=0.6, color='green')
        plt.xlabel('Current Run Rate')
        plt.ylabel('Total Score')
        plt.title('Run Rate vs Total Score')
        plt.grid(True, alpha=0.3)
    
    # 3. Wickets vs Total score
    plt.subplot(1, 3, 3)
    if 'wickets' in data.columns and 'total' in data.columns:
        plt.scatter(data['wickets'], data['total'], alpha=0.6, color='red')
        plt.xlabel('Wickets Lost')
        plt.ylabel('Total Score')
        plt.title('Wickets vs Total Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cricket_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Additional analysis plots saved as cricket_analysis_plots.png")
    
    # 4. Model comparison bar chart
    if results:
        plt.figure(figsize=(10, 6))
        
        models = list(results.keys())
        mae_scores = [results[model]['MAE'] for model in models]
        rmse_scores = [results[model]['RMSE'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8)
        plt.bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Error Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, [model.replace('_', ' ').title() for model in models])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Model comparison plot saved as model_comparison.png")

def demo_live_predictions(predictor):
    """Demonstrate live score predictions"""
    print("Demonstrating live score predictions...")
    
    # Sample match scenarios
    scenarios = [
        {"overs": 5.0, "runs": 45, "wickets": 1, "description": "Powerplay - Good start"},
        {"overs": 10.0, "runs": 85, "wickets": 3, "description": "Middle overs - Average"},
        {"overs": 15.0, "runs": 130, "wickets": 4, "description": "Death overs approach"},
        {"overs": 18.0, "runs": 160, "wickets": 6, "description": "Death overs - Pressure"},
    ]
    
    print("\nLive Prediction Scenarios:")
    print("-" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            predicted_total = predictor.predict_live_score(
                scenario["overs"], 
                scenario["runs"], 
                scenario["wickets"]
            )
            
            current_rr = scenario["runs"] / scenario["overs"]
            required_rr = (predicted_total - scenario["runs"]) / (20 - scenario["overs"])
            
            print(f"\nScenario {i}: {scenario['description']}")
            print(f"Current: {scenario['overs']} overs, {scenario['runs']}/{scenario['wickets']}")
            print(f"Current RR: {current_rr:.2f}")
            print(f"Predicted Total: {predicted_total:.0f}")
            print(f"Required RR: {required_rr:.2f}")
            
        except Exception as e:
            print(f"Error in scenario {i}: {e}")

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run main application
    main()