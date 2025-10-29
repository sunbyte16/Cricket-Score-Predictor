import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class CricketDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """Preprocess cricket data for machine learning"""
        print("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Create new features
        data = self._create_features(data)
        
        # Encode categorical variables
        data = self._encode_categorical_features(data)
        
        # Select features for modeling
        feature_cols = self._select_features(data)
        
        print(f"Preprocessing completed. Features selected: {len(feature_cols)}")
        return data, feature_cols
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def _create_features(self, data):
        """Create new features from existing data"""
        print("Creating new features...")
        
        # Current run rate
        if 'runs' in data.columns and 'overs' in data.columns:
            data['current_run_rate'] = data['runs'] / (data['overs'] + 0.1)  # Avoid division by zero
        
        # Balls remaining
        if 'overs' in data.columns:
            data['balls_remaining'] = (20 - data['overs']) * 6
            data['overs_remaining'] = 20 - data['overs']
        
        # Wickets remaining
        if 'wickets' in data.columns:
            data['wickets_remaining'] = 10 - data['wickets']
        
        # Required run rate (if chasing)
        if 'runs' in data.columns and 'balls_remaining' in data.columns:
            # Assuming target is available or can be estimated
            data['required_run_rate'] = np.where(
                data['balls_remaining'] > 0,
                (200 - data['runs']) / (data['balls_remaining'] / 6),  # Assuming target of 200
                0
            )
        
        # Recent form indicators
        if 'runs_last_5' in data.columns and 'wickets_last_5' in data.columns:
            data['recent_run_rate'] = data['runs_last_5'] / 5
            data['recent_wicket_rate'] = data['wickets_last_5'] / 5
        
        # Powerplay indicator
        if 'overs' in data.columns:
            data['is_powerplay'] = (data['overs'] <= 6).astype(int)
            data['is_death_overs'] = (data['overs'] >= 16).astype(int)
            data['is_middle_overs'] = ((data['overs'] > 6) & (data['overs'] < 16)).astype(int)
        
        return data
    
    def _encode_categorical_features(self, data):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = ['venue', 'batting_team', 'bowling_team', 'striker', 'non_striker']
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data
    
    def _select_features(self, data):
        """Select relevant features for modeling"""
        print("Selecting features...")
        
        # Define potential feature columns
        potential_features = [
            'overs', 'runs', 'wickets', 'current_run_rate', 'balls_remaining',
            'wickets_remaining', 'required_run_rate', 'recent_run_rate',
            'recent_wicket_rate', 'is_powerplay', 'is_death_overs', 'is_middle_overs',
            'venue_encoded', 'batting_team_encoded', 'bowling_team_encoded',
            'runs_last_5', 'wickets_last_5'
        ]
        
        # Select only features that exist in the data
        feature_cols = [col for col in potential_features if col in data.columns]
        self.feature_columns = feature_cols
        
        return feature_cols
    
    def prepare_train_test_split(self, data, feature_cols, target_col='total', test_size=0.2):
        """Prepare train-test split"""
        print("Preparing train-test split...")
        
        X = data[feature_cols]
        y = data[target_col]
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_names(self):
        """Get feature names for importance plotting"""
        return self.feature_columns