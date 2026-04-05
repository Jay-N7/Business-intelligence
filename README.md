# Sales Prediction Project 

This project analyzes sales data and creates machine learning models to predict future sales trends.

## Features
- Data analysis and visualization
- Interactive dashboard (web app)
- Machine learning predictions (linear regression, random forest)
- Flask web application

## Basic workflow
- Install and setup miniconda for the conda enviroments.
- Install all the required tools (pandas numpy matplotlib plotly scikit-learn jupyter flask,kaggle seaborn plotly-dash)
- Set up kaggle API (download the required csv file with the sample data). Amazon sales rport in my case
- Start coding on jupytr notebook(sales_analysis.ipynb), different cells for different tasks like :
    - Importing rquired libraries
    - Load the data and gets its info
    - Handle missing values and clean the data
    - Plot a basic graph for sales over time (visualization)
    - Prepare data for prediction (loading into ariables for models)
    - Train prediction models (hav split data into traning and testing)
    - Tried 2 models: linear regression and random forest
    - Plot actual vs predicted values 
    - Save and use the btter model (linear regression for this data)
- Write bsic python scripts to process data,predict and visualse
- Create a flask web app (app.py)
- Create HTML templates for th flask web app
- App has Home pag and dashboard (graphs and predictor)
- Use basic css templates for styling 
- Testing and debugging 
- Ran the web app succesfully

# STRUCTURE IN DETAIL :
sales_prediction_project/
├── app.py                           # Main Flask application
├── data/                           # Data directory
│   ├── Amazon Sale Report.csv           # Downloaded dataset
│   └── unlock-profits-with-e-commerce-sales-data.zip
├── models/                         # Saved models directory
│   ├── random_forest_model.pkl    # Trained model
│   └── Amazon_data_columns.json       # Feature columns metadata
├── notebooks/                      # Jupyter notebooks
│   └── sales_analysis.ipynb       # Main analysis notebook
├── output/                         # Generated outputs
│   ├── sales_over_time.html       # Generated plots
│   ├── monthly_sales.html
│   ├── sales_distribution.html
│   └── predictions_comparison.html
├── src/                           # Source code modules
│   |             
│   ├── data_processor.py         # Data processing class
│   ├── visualizer.py             # Visualization class
│   └── predictor.py              # Prediction model class
├── static/                        # Static files for Flask
│   └── style.css                 # CSS styling
└── templates/                     # HTML templates
    ├── base.html                 # Base template
    ├── index.html                # Home page
    └── dashboard.html            # Dashboard page
```

