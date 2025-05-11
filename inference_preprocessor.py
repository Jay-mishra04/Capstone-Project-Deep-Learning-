import pandas as pd
import numpy as np
import joblib

class CSATPreprocessorForInference:
    def __init__(self):
        self.label_encoder = joblib.load('label_encoder.pkl')
        self.rs = joblib.load('robust_scaler.pkl')
        self.ss = joblib.load('standard_scaler.pkl')
        self.feature_columns = joblib.load('feature_columns.pkl')

    def preprocess(self, df):
        try:
            # Clean and convert datetime fields
            df['order_date_time'] = pd.to_datetime(df['order_date_time'].str.strip(), format="%d/%m/%Y %H:%M")
            df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'].str.strip(), format="%d/%m/%Y %H:%M")
            df['issue_responded'] = pd.to_datetime(df['issue_responded'].str.strip(), format="%d/%m/%Y %H:%M")
            df['Survey_response_Date'] = pd.to_datetime(df['Survey_response_Date'].str.strip(), format="%d/%b/%Y")
            print("[✓] Datetime conversion completed.")
        except Exception as e:
            print("[✗] Error in datetime conversion:", e)

        # Fill missing values
        df['order_date_time'] = df['order_date_time'].fillna(df['order_date_time'].median())
        df['Product_category'] = df['Product_category'].fillna('Unknown')
        df['Item_price'] = df.groupby('Product_category')['Item_price'].transform(lambda x: x.fillna(x.median()))

        # Feature engineering
        df['response_time'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
        df['order_issue_gap'] = (df['Issue_reported at'] - df['order_date_time']).dt.days
        df['survey_delay'] = (df['Survey_response_Date'].dt.date - df['issue_responded'].dt.date).apply(lambda x: x.days)
        df = df[(df['order_issue_gap'] >= 0) | (df['order_issue_gap'].isnull())]
        df = df[df['response_time'] >= 0]
        df = df[df['survey_delay'] >= 0]

        df['issue_reported_hour'] = df['Issue_reported at'].dt.hour
        df['issue_reported_dayofweek'] = df['Issue_reported at'].dt.dayofweek
        df['is_weekend'] = df['Issue_reported at'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        # Log transform
        df['Item_price_log'] = np.log1p(df['Item_price'])
        df['order_issue_gap_log'] = np.log1p(df['order_issue_gap'])
        df['response_time_log'] = np.log1p(df['response_time'])

        # Scaling
        robust_cols = ['Item_price_log', 'order_issue_gap_log', 'response_time_log',
                       'Supervisor_csat_score', 'Agent_case_count', 'Agent_csat_score']
        standard_cols = ['issue_reported_hour', 'issue_reported_dayofweek', 'is_weekend', 'Supervisor_case_count']
        df[robust_cols] = self.rs.transform(df[robust_cols])
        df[standard_cols] = self.ss.transform(df[standard_cols])

        # Drop unneeded columns
        drop_cols = ['Agent_name', 'survey_delay', 'order_issue_gap', 'response_time', 'Item_price',
                     'Issue_reported at', 'issue_responded', 'Survey_response_Date', 'connected_handling_time',
                     'Product_category', 'Unique id', 'order_date_time', 'Customer Remarks', 'Customer_City',
                     'Order_id', 'Supervisor']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Encoding
        df['Sub_category_encoded'] = self.label_encoder.transform(df['Sub-category'])
        df.drop(columns=['Sub-category'], inplace=True)
        df = pd.get_dummies(df, drop_first=True, dtype='int32')

        # Align with training columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns + ['Sub_category_encoded']]

        return df
