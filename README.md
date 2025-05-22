# Shoe Sales Tracker and Forecasting System

A Streamlit web application for tracking daily shoe sales across multiple institutions and forecasting future sales based on historical data.

## Live Demo

**Live Application:** [https://ggcintern.streamlit.app/](https://ggcintern.streamlit.app/)

## Features

### Daily Sales Tracking
- Record sales data by size (4-15) for different institutions
- Manage data entries with delete functionality
- Compare sales data across selected dates
- Visualize size distribution with interactive charts
- Auto-calculate total sales
- Save and load tracking data from CSV

### Sales Forecasting
- Load historical sales data for analysis
- Generate forecasts using Prophet and Simple Moving Average
- View size distribution across different years
- Interactive visualization of sales trends and forecasts
- Download forecasted data for future planning

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd hackathon
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Ensure you have the following data files:
- `daily_sales.csv` (will be created automatically if not present)
- `actual_sales.csv` (required for forecasting feature)

## Usage

Run the application with:
```
streamlit run app.py
```

### Navigation
Use the sidebar to switch between application modes:
- **Daily Sales Tracking**: Record and analyze daily sales
- **Sales Forecasting**: Generate and visualize sales forecasts

### Data Management
- Record new sales entries using the form interface
- Delete unwanted entries by selecting them in the data editor
- Compare historical data with interactive charts

## Deployment

This application is already deployed at [https://ggcintern.streamlit.app/](https://ggcintern.streamlit.app/)


## Technologies Used
- Streamlit: Web application framework
- Pandas: Data manipulation and analysis
- Prophet: Time series forecasting
- Plotly: Interactive data visualization 
