import pandas as pd
import numpy as np

def load_cricket_data():
    """Load T20I cricket dataset from Kaggle or create sample data"""
    try:
        # Try to import kagglehub if available
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        print("Attempting to load dataset from Kaggle...")
        
        # Set the path to the file you'd like to load
        file_path = ""
        
        # Load the latest version
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sohail945/t20i-cricket-score-prediction",
            file_path,
            # Provide any additional arguments like 
            # sql_query or pandas_kwargs. See the 
            # documentation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("First 5 records:")
        print(df.head())
        print("\nDataset info:")
        print(df.info())
        print("\nColumn names:")
        print(df.columns.tolist())
        
        return df
        
    except ImportError:
        print("kagglehub not available. Creating sample dataset for demonstration...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample dataset for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample cricket data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'venue': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'], n_samples),
        'batting_team': np.random.choice(['Team_A', 'Team_B', 'Team_C', 'Team_D'], n_samples),
        'bowling_team': np.random.choice(['Team_A', 'Team_B', 'Team_C', 'Team_D'], n_samples),
        'overs': np.random.uniform(0.1, 20.0, n_samples),
        'runs': np.random.randint(0, 250, n_samples),
        'wickets': np.random.randint(0, 10, n_samples),
        'runs_last_5': np.random.randint(20, 80, n_samples),
        'wickets_last_5': np.random.randint(0, 5, n_samples),
        'striker': np.random.choice(['Player_1', 'Player_2', 'Player_3', 'Player_4'], n_samples),
        'non_striker': np.random.choice(['Player_1', 'Player_2', 'Player_3', 'Player_4'], n_samples),
        'total': np.random.randint(120, 220, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Make data more realistic
    df['run_rate'] = df['runs'] / df['overs']
    df['balls_left'] = (20 - df['overs']) * 6
    df['wickets_left'] = 10 - df['wickets']
    
    print("Sample dataset created for demonstration")
    print(f"Dataset shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    df = load_cricket_data()