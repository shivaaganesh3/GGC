import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import datetime
from pathlib import Path
import traceback

# ==============================================
# Forecasting Section Components
# ==============================================

@st.cache_data
def load_forecast_data():
    try:
        df = pd.read_csv('actual_sales.csv')
        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'], format='%d-%m-%Y')
        
        # Convert to long format
        size_cols = [col for col in df.columns if col.startswith('Size_')]
        df_long = df.melt(
            id_vars=['Institution', 'Sale_Date', 'Total_Sales'],
            value_vars=size_cols,
            var_name='Size',
            value_name='Sales'
        )
        df_long['Size'] = df_long['Size'].str.replace('Size_', '').astype(int)
        
        return df, df_long
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def prophet_forecast(df, institution):
    try:
        model = Prophet(
            yearly_seasonality=False,
            changepoint_prior_scale=0.001,
            n_changepoints=min(2, len(df)-1)
        )
        model.fit(df)
        future_date = df['ds'].max() + pd.DateOffset(years=1)
        future = pd.DataFrame({'ds': [future_date]})
        forecast = model.predict(future)
        return round(forecast['yhat'].iloc[0])
    except Exception as e:
        st.error(f"Prophet forecast failed: {str(e)}")
        return None

def forecasting_main():
    st.title("📊 Institution Sales Forecasting System")
    
    # Load data
    df_wide, df_long = load_forecast_data()
    if df_wide is None:
        return

    # Institution selection
    institutions = df_wide['Institution'].unique()
    selected_inst = st.selectbox("Select Institution:", institutions)

    # Filter institution data
    inst_data = df_wide[df_wide['Institution'] == selected_inst]
    if len(inst_data) < 2:
        st.warning("Insufficient data for forecasting (need at least 2 years)")
        return

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Institutional Sales Data")
        st.dataframe(inst_data)

    # Forecast section
    st.header("Forecast Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Sales Forecast")
        # Prophet forecast
        prophet_df = inst_data[['Sale_Date', 'Total_Sales']].rename(
            columns={'Sale_Date': 'ds', 'Total_Sales': 'y'}
        )
        prophet_total = prophet_forecast(prophet_df, selected_inst)
        
        # SMA forecast
        sma_window = st.slider("SMA Window (years)", 1, 3, 2)
        sma_total = round(inst_data['Total_Sales'].tail(sma_window).mean())

    # Size distribution forecast
    with col2:
        st.subheader("Size Distribution")
        size_ratios = inst_data.filter(regex='Size_').div(inst_data['Total_Sales'], axis=0).mean()
        selected_sizes = st.multiselect(
            "Select Sizes to Display:",
            options=[int(s.replace('Size_', '')) for s in size_ratios.index],
            default=[8, 9, 10]
        )

    # Visualization
    st.header("Results Visualization")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Total Sales", "Size Distribution", "Forecast Details"])

    with tab1:
        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(
            x=inst_data['Sale_Date'],
            y=inst_data['Total_Sales'],
            mode='lines+markers',
            name='Historical'
        ))
        
        if prophet_total:
            future_date = inst_data['Sale_Date'].max() + pd.DateOffset(years=1)
            fig_total.add_trace(go.Scatter(
                x=[future_date],
                y=[prophet_total],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Prophet Forecast'
            ))
        
        fig_total.add_trace(go.Scatter(
            x=[future_date],
            y=[sma_total],
            mode='markers',
            marker=dict(color='green', size=12),
            name=f'{sma_window}-Year SMA'
        ))
        
        fig_total.update_layout(
            title=f'{selected_inst} Total Sales Forecast',
            xaxis_title='Year',
            yaxis_title='Total Sales'
        )
        st.plotly_chart(fig_total)

    with tab2:
        if selected_sizes:
            fig_sizes = go.Figure()
            for size in selected_sizes:
                size_data = df_long[(df_long['Institution'] == selected_inst) & 
                                  (df_long['Size'] == size)]
                fig_sizes.add_trace(go.Scatter(
                    x=size_data['Sale_Date'],
                    y=size_data['Sales'],
                    mode='lines+markers',
                    name=f'Size {size}'
                ))
            
            fig_sizes.update_layout(
                title=f'{selected_inst} Size-wise Sales Trend',
                xaxis_title='Year',
                yaxis_title='Sales Quantity'
            )
            st.plotly_chart(fig_sizes)
        else:
            st.info("Select sizes to view their trends")

    with tab3:
        if prophet_total:
            st.subheader("Forecast Details")
            
            # Calculate size distributions
            prophet_sizes = (size_ratios * prophet_total).round().astype(int)
            sma_sizes = (size_ratios * sma_total).round().astype(int)
            
            # Create results dataframe
            forecast_df = pd.DataFrame({
                'Size': [int(s.replace('Size_', '')) for s in size_ratios.index],
                'Prophet Forecast': prophet_sizes.values,
                'SMA Forecast': sma_sizes.values
            }).set_index('Size')
            
            st.dataframe(forecast_df.style.highlight_max(axis=1, color='#90EE90'),
                        use_container_width=True)
            
            st.download_button(
                label="Download Forecasts",
                data=forecast_df.to_csv().encode('utf-8'),
                file_name=f'{selected_inst}_forecast.csv',
                mime='text/csv'
            )
        else:
            st.warning("No forecast available to show details")

# ==============================================
# Tracking Section Components
# ==============================================

TRACKING_DATA_FILE = Path("daily_sales.csv")
TRACKING_SIZE_RANGE = range(4, 16)
TRACKING_INSTITUTIONS = ["Retail Store A", "Online Store", "Outlet Store"]

def init_tracking_dataframe():
    """Initialize empty DataFrame with correct structure"""
    return pd.DataFrame(columns=["Date", "Institution"] + 
                       [f"Size_{i}" for i in TRACKING_SIZE_RANGE] + 
                       ["Total"])

@st.cache_resource(show_spinner=False)
def load_tracking_data():
    """Load or create sales data with error handling"""
    try:
        if TRACKING_DATA_FILE.exists():
            df = pd.read_csv(TRACKING_DATA_FILE)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df, "Data loaded successfully!", None
        else:
            init_tracking_dataframe().to_csv(TRACKING_DATA_FILE, index=False)
            return init_tracking_dataframe(), "New data file created!", None
    except Exception as e:
        error_msg = f"Data loading failed: {str(e)}"
        stack_trace = traceback.format_exc()
        return init_tracking_dataframe(), None, (error_msg, stack_trace)

def save_tracking_data(df):
    """Save data with error handling"""
    try:
        df.to_csv(TRACKING_DATA_FILE, index=False)
        st.cache_resource.clear()
        st.toast("Data saved successfully!")
    except Exception as e:
        st.error(f"Data saving failed: {str(e)}")
        st.code(traceback.format_exc())

def tracking_main():
    st.title("👟 Daily Shoe Sales Tracker")
    
    # Initialize session state
    if "sales_data" not in st.session_state:
        data_result, success_msg, error_info = load_tracking_data()
        st.session_state.sales_data = data_result
        if success_msg:
            st.toast(success_msg)
        if error_info:
            st.error(error_info[0])
            st.code(error_info[1])
    
    # Debug panel
    with st.expander("Debug Tools"):
        if st.button("Reload Data"):
            data_result, success_msg, error_info = load_tracking_data()
            st.session_state.sales_data = data_result
            if success_msg:
                st.toast(success_msg)
            if error_info:
                st.error(error_info[0])
                st.code(error_info[1])
        if st.button("Clear Cache"):
            st.cache_resource.clear()
        st.write("Data shape:", st.session_state.sales_data.shape)
    
    # Date selection
    current_date = st.date_input("Select Date", datetime.date.today())
    
    # Institution selection
    institution = st.selectbox("Select Institution", TRACKING_INSTITUTIONS)
    
    # Sales entry form
    with st.form("sales_form"):
        st.subheader(f"Sales Entry for {current_date}")
        
        # Size inputs
        size_values = {
            size: st.number_input(
                f"Size {size}", 
                min_value=0, 
                value=0,
                key=f"size_{size}_{current_date}"
            )
            for size in TRACKING_SIZE_RANGE
        }
        
        # Calculate total
        total = sum(size_values.values())
        st.metric("Total Pairs Sold", total)
        
        if st.form_submit_button("Save Entry"):
            try:
                # Create new entry
                new_entry = {
                    "Date": current_date,
                    "Institution": institution,
                    **{f"Size_{k}": v for k, v in size_values.items()},
                    "Total": total
                }
                
                # Update DataFrame
                updated_df = pd.concat([
                    st.session_state.sales_data,
                    pd.DataFrame([new_entry])
                ], ignore_index=True)
                
                # Remove duplicates
                updated_df = updated_df.drop_duplicates(
                    subset=["Date", "Institution"], 
                    keep="last"
                )
                
                # Update state and save
                st.session_state.sales_data = updated_df
                save_tracking_data(updated_df)
                
            except Exception as e:
                st.error(f"Entry failed: {str(e)}")
                st.code(traceback.format_exc())
    if not st.session_state.sales_data.empty:
        st.header("📋 Entry Management")
        st.subheader("Delete Existing Entries")
        
        # Create editable DataFrame with checkboxes
        df = st.session_state.sales_data.copy()
        df.insert(0, "Select", False)
        
        edited_df = st.data_editor(
            df,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Delete?",
                    help="Select entries to delete"
                ),
                "Date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD",
                    required=True
                ),
                "Total": st.column_config.NumberColumn(
                    "Total Sales",
                    format="%d pairs"
                )
            },
            disabled=["Date", "Institution"] + [f"Size_{i}" for i in TRACKING_SIZE_RANGE] + ["Total"],
            use_container_width=True
        )
        
        if st.button("🗑️ Delete Selected Entries", type="primary"):
            # Find selected rows
            selected_indices = edited_df.index[edited_df["Select"]].tolist()
            
            if selected_indices:
                # Filter out selected rows
                updated_df = st.session_state.sales_data.drop(selected_indices)
                
                # Update state and save
                st.session_state.sales_data = updated_df
                save_tracking_data(updated_df)
                st.toast(f"Deleted {len(selected_indices)} entries")
                st.rerun()
            else:
                st.warning("No entries selected for deletion")

    # Data display section
    if not st.session_state.sales_data.empty:
        st.header("Sales Analysis")
        
        # Filter data
        filtered = st.session_state.sales_data[
            (st.session_state.sales_data["Institution"] == institution)
        ]
        
        # Date selector for comparison
        dates = filtered["Date"].unique()
        selected_dates = st.multiselect(
            "Compare Dates",
            options=dates,
            default=dates[-2:] if len(dates) >=2 else dates,
            format_func=lambda d: d.strftime("%Y-%m-%d")
        )
        
        # Display comparison
        if selected_dates:
            col1, col2 = st.columns(2)
            comparison_data = filtered[filtered["Date"].isin(selected_dates)]
            
            with col1:
                st.subheader("Size Distribution")
                st.bar_chart(
                    comparison_data.set_index("Date")[[f"Size_{i}" for i in TRACKING_SIZE_RANGE]]
                )
            
            with col2:
                st.subheader("Numerical Data")
                st.dataframe(
                    comparison_data.set_index("Date").sort_index(ascending=False),
                    use_container_width=True
                )
            
            st.subheader("Trend Analysis")
            trend_data = filtered[filtered["Date"].isin(selected_dates)]
            st.line_chart(trend_data.set_index("Date")["Total"])
    else:
        st.info("No sales data recorded yet")

# ==============================================
# Main Application with Navigation
# ==============================================

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Application Mode", 
                              ["Daily Sales Tracking", "Sales Forecasting"])
    
    if app_mode == "Daily Sales Tracking":
        tracking_main()
    else:
        forecasting_main()

if __name__ == "__main__":
    main()
