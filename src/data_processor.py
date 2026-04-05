import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_data(self, file_path):
        """Load data from CSV file"""
        self.df = pd.read_csv(
            '/home/archer/Nuclear-Codes/Internship/Week2/sales_prediction_project/data/Amazon Sale Report.csv',
            low_memory=False
        )

        self.df.columns = self.df.columns.str.strip().str.lower()

        print("Columns:", self.df.columns.tolist())
        if 'date' in self.df.columns:
            print("First few values in date column:", self.df['date'].head())

        return self.df

    def clean_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.df = self.df.dropna()

        self.df.columns = self.df.columns.str.strip().str.lower()

        if 'amount' in self.df.columns:
            self.df.rename(columns={'amount': 'sales'}, inplace=True)

        if 'sales' in self.df.columns:
            self.df['sales'] = self.df['sales'].replace('[₹,]', '', regex=True).astype(float)


        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], format='%m-%d-%y', errors='coerce')
            print("Parsed 'date' column. Type:", self.df['date'].dtype)

        return self.df

    def create_features(self, date_col='date', target_col='sales'):
        """Create features for machine learning"""
        if self.df is None or date_col not in self.df.columns:
            raise ValueError(f"'{date_col}' column not found in the DataFrame.")

        self.df['year'] = self.df[date_col].dt.year
        self.df['month'] = self.df[date_col].dt.month
        self.df['day'] = self.df[date_col].dt.day
        self.df['day_of_week'] = self.df[date_col].dt.dayofweek

        self.df.loc[:, f'{target_col}_lag_1'] = self.df[target_col].shift(1)
        self.df.loc[:, f'{target_col}_lag_7'] = self.df[target_col].shift(7)
        self.df.loc[:, f'{target_col}_rolling_7'] = self.df[target_col].rolling(window=7).mean()


        return self.df.dropna()

