from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from inference_preprocessor import CSATPreprocessorForInference

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = tf.keras.models.load_model('csat_model.keras')
preprocessor = CSATPreprocessorForInference()

# Load agent and supervisor stats
agent_stats = pd.read_csv('agent_stats.csv')
supervisor_stats = pd.read_csv('supervisor_stats.csv')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Collect and clean form inputs
        form_data = {key: request.form[key].strip() for key in request.form}
        df = pd.DataFrame([form_data])

        # Step 2: Parse numeric fields
        df['Item_price'] = pd.to_numeric(df['Item_price'])
        df['connected_handling_time'] = pd.to_numeric(df['connected_handling_time'])

        # Step 3: Fill in CSAT stats from lookup
        agent_name = df.loc[0, 'Agent_name']
        supervisor_name = df.loc[0, 'Supervisor']

        agent_row = agent_stats[agent_stats['Agent_name'] == agent_name]
        supervisor_row = supervisor_stats[supervisor_stats['Supervisor'] == supervisor_name]

        df['Agent_case_count'] = agent_row['Agent_case_count'].values[0] 
        df['Agent_csat_score'] = agent_row['Agent_csat_score'].values[0] 
        df['Supervisor_case_count'] = supervisor_row['Supervisor_case_count'].values[0] 
        df['Supervisor_csat_score'] = supervisor_row['Supervisor_csat_score'].values[0] 

        # Step 4: Add other required fields
        df['Customer Remarks'] = ''
        df['Customer_City'] = 'Delhi'
        df['Order_id'] = 123456
        df['Unique id'] = 10001

        # Step 5: Strip datetime inputs
        for col in ['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date']:
            df[col] = df[col].str.strip()

        # Step 6: Preprocess the cleaned DataFrame
        processed_df = preprocessor.preprocess(df)

        # Step 7: Prepare model inputs
        subcat_input = processed_df['Sub_category_encoded'].values
        other_input = processed_df.drop(columns=['Sub_category_encoded']).values

        # Step 8: Predict
        prediction = model.predict({'subcat_input': subcat_input, 'other_features': other_input})
        predicted_score = np.argmax(prediction, axis=1)[0] + 1  # convert back to 1–5 CSAT

        return render_template('form.html', prediction=f"✅ Predicted CSAT Score: {predicted_score}")

    except Exception as e:
        return render_template('form.html', prediction=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
