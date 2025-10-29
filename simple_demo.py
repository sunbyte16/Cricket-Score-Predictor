#!/usr/bin/env python3
"""
Simple Cricket Score Predictor Demo
Works with minimal dependencies and sample data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def create_cricket_dataset(n_samples=2000):
    """Create a realistic cricket dataset"""
    np.random.seed(42)
    
    # Generate realistic cricket data
    venues = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Jaipur']
    teams = ['MI', 'CSK', 'RCB', 'KKR', 'DC', 'RR', 'PBKS', 'SRH']
    
    data = []
    
    for i in range(n_samples):
        # Basic match info
        venue = np.random.choice(venues)
        batting_team = np.random.choice(teams)
        bowling_team = np.random.choice([t for t in teams if t != batting_team])
        
        # Current match situation
        overs = np.random.uniform(1.0, 19.9)
        wickets = np.random.randint(0, min(10, int(overs/2) + 3))
        
        # Generate realistic runs based on overs and wickets
        base_runs = overs * np.random.uniform(6, 12)
        wicket_penalty = wickets * np.random.uniform(0, 5)
        runs = max(0, int(base_runs - wicket_penalty + np.random.normal(0, 10)))
        
        # Recent form (last 5 overs)
        runs_last_5 = np.random.randint(20, 80)
        wickets_last_5 = np.random.randint(0, max(1, min(3, wickets + 1)))
        
        # Generate realistic final total
        remaining_overs = 20 - overs
        remaining_wickets = 10 - wickets
        
        # Predict final score based on current situation
        current_rr = runs / overs if overs > 0 else 6
        projected_runs = runs + (remaining_overs * current_rr * (remaining_wickets/10))
        
        # Add some randomness and constraints
        final_total = max(runs + 10, min(250, int(projected_runs + np.random.normal(0, 20))))
        
        data.append({
            'venue': venue,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'overs': round(overs, 1),
            'runs': runs,
            'wickets': wickets,
            'runs_last_5': runs_last_5,
            'wickets_last_5': wickets_last_5,
            'total': final_total
        })
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the cricket data"""
    print("Preprocessing data...")
    
    # Create new features
    df['current_run_rate'] = df['runs'] / (df['overs'] + 0.1)
    df['balls_remaining'] = (20 - df['overs']) * 6
    df['wickets_remaining'] = 10 - df['wickets']
    df['recent_run_rate'] = df['runs_last_5'] / 5
    df['is_powerplay'] = (df['overs'] <= 6).astype(int)
    df['is_death_overs'] = (df['overs'] >= 16).astype(int)
    
    # Encode categorical variables
    le_venue = LabelEncoder()
    le_batting = LabelEncoder()
    le_bowling = LabelEncoder()
    
    df['venue_encoded'] = le_venue.fit_transform(df['venue'])
    df['batting_team_encoded'] = le_batting.fit_transform(df['batting_team'])
    df['bowling_team_encoded'] = le_bowling.fit_transform(df['bowling_team'])
    
    # Select features
    feature_cols = [
        'overs', 'runs', 'wickets', 'current_run_rate', 'balls_remaining',
        'wickets_remaining', 'recent_run_rate', 'is_powerplay', 'is_death_overs',
        'venue_encoded', 'batting_team_encoded', 'bowling_team_encoded',
        'runs_last_5', 'wickets_last_5'
    ]
    
    return df, feature_cols

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': y_pred
        }
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
    print(f"\nBest model: {best_model_name}")
    
    return results, best_model_name

def plot_results(y_test, results):
    """Plot actual vs predicted results"""
    try:
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            ax.scatter(y_test, result['predictions'], alpha=0.6)
            
            # Perfect prediction line
            min_val = min(min(y_test), min(result['predictions']))
            max_val = max(max(y_test), max(result['predictions']))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f'{name}\nR² = {result["R2"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'simple_predictions.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")

def predict_live_score(model, current_overs, current_runs, current_wickets):
    """Predict final score based on current situation"""
    # Create feature vector
    current_rr = current_runs / current_overs if current_overs > 0 else 6
    balls_remaining = (20 - current_overs) * 6
    wickets_remaining = 10 - current_wickets
    is_powerplay = 1 if current_overs <= 6 else 0
    is_death_overs = 1 if current_overs >= 16 else 0
    
    # Simplified feature vector (using averages for encoded features)
    features = np.array([[
        current_overs, current_runs, current_wickets, current_rr,
        balls_remaining, wickets_remaining, 40, is_powerplay, is_death_overs,
        2, 3, 4, 40, 1  # Average values for encoded features
    ]])
    
    prediction = model.predict(features)[0]
    return max(current_runs, prediction)

def main():
    """Main function"""
    print("=" * 60)
    print("SIMPLE CRICKET SCORE PREDICTOR DEMO")
    print("=" * 60)
    
    # Create dataset
    print("\n1. Creating sample cricket dataset...")
    df = create_cricket_dataset(2000)
    print(f"Dataset created with {len(df)} samples")
    print(f"Score range: {df['total'].min()} - {df['total'].max()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    df, feature_cols = preprocess_data(df)
    
    # Prepare train-test split
    print("\n3. Preparing train-test split...")
    X = df[feature_cols]
    y = df['total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    print("\n4. Training and evaluating models...")
    results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot results
    print("\n5. Creating visualizations...")
    plot_results(y_test, results)
    
    # Demo predictions
    print("\n6. Live prediction demo...")
    best_model = results[best_model_name]['model']
    
    scenarios = [
        (6.0, 55, 2, "End of powerplay - good start"),
        (10.0, 85, 3, "Middle overs - steady progress"),
        (15.0, 130, 5, "Death overs approach - need acceleration"),
        (18.0, 160, 7, "Final overs - pressure situation")
    ]
    
    print("\nLive Prediction Scenarios:")
    print("-" * 50)
    
    for overs, runs, wickets, description in scenarios:
        prediction = predict_live_score(best_model, overs, runs, wickets)
        current_rr = runs / overs
        required_rr = (prediction - runs) / (20 - overs) if overs < 20 else 0
        
        print(f"\n{description}")
        print(f"Current: {overs} overs, {runs}/{wickets}")
        print(f"Current RR: {current_rr:.2f}")
        print(f"Predicted Total: {prediction:.0f}")
        print(f"Required RR: {required_rr:.2f}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()