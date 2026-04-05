import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, df):
        self.df = df
    
    def plot_sales_over_time(self, date_col='date', sales_col='sales', save_path=None):
        """Create sales over time plot"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df[date_col],
            y=self.df[sales_col],
            mode='lines+markers',
            name='Sales',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Sales Over Time',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_monthly_sales(self, date_col='date', sales_col='sales', save_path=None):
        """Create monthly sales analysis"""
        df_monthly = self.df.copy()
        df_monthly['month'] = df_monthly[date_col].dt.to_period('M')
        monthly_sales = df_monthly.groupby('month')[sales_col].sum().reset_index()
        monthly_sales['month'] = monthly_sales['month'].astype(str)
        
        fig = px.bar(monthly_sales, x='month', y=sales_col, 
                     title='Monthly Sales Analysis')
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_predictions(self, actual, predicted, save_path=None):
        """Plot actual vs predicted values"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Sales',
            xaxis_title='Actual Sales',
            yaxis_title='Predicted Sales',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

