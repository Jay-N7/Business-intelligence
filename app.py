from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import plotly
import plotly.graph_objects as go
from src.data_processor import DataProcessor
from src.visualizer import Visualizer
from src.predictor import SalesPredictor

app = Flask(__name__)

processor = DataProcessor()
df = processor.load_data('data/Amazon Sale Report.csv')  # Replace with actual file
df_clean = processor.clean_data()
df_features = processor.create_features()

# Load trained model
predictor = SalesPredictor()
predictor.load_model('models/amazon_linear_regression_model.pkl', 'models/amazon_feature_columns.json')

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/sales_over_time')
def sales_over_time():
    """API endpoint for sales over time chart"""
    visualizer = Visualizer(df_clean)
    fig = visualizer.plot_sales_over_time(date_col='date', sales_col='sales')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/api/monthly_sales')
def monthly_sales():
    """API endpoint for monthly sales chart"""
    visualizer = Visualizer(df_clean)
    fig = visualizer.plot_monthly_sales(date_col='date', sales_col='sales')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/api/predict', methods=['POST'])
def predict_sales():
    """API endpoint for sales prediction"""
    try:
        data = request.json
        features = pd.DataFrame([data])
        prediction = predictor.predict(features)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.route('/dashboard')
def dashboard():
    """Dashboard page with all visualizations"""
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

