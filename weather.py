import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .box {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        width: 200px;
        height: 230px;
    }
    .box h2 {
        color: #333;
        margin-bottom: 10px;
    }
    .box p {
        font-size: 1.5em;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

def create_bar_plots(df):
    for column in df.columns:
        if column not in ['Tillage Operation', 'Farm Name']:
            st.subheader(f"{column}")
            total_count = df.shape[0] - 1
            done_count = df[df[column].isin(['Done', 'Done early', 'Done on time'])].shape[0]
            df_plot = pd.DataFrame({
                'Status': ['Total', 'Done and Pop Followed'],
                'Count': [total_count, done_count]
            })
            colors = ['blue', 'green']  # Set the color for "Done and Pop Followed" as green
            fig = px.bar(df_plot, x='Count', y='Status', orientation='h', text='Count', width=800, height=200, color='Status', color_discrete_sequence=colors)
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                xaxis_title="Count",
                yaxis_title="Status",
                title=f"Counts for {column}",
                showlegend=False
            )
            st.plotly_chart(fig)

def create_line_chart(df):
    df['Date'] = pd.to_datetime(df['Date'])    
    last_week = datetime.now() - timedelta(days=7)
    df_filtered = df[df['Date'] >= last_week]
    df_counts = df_filtered.groupby('FarmName')['Activity'].nunique().reset_index()
    fig = px.line(df_counts, x='FarmName', y='Activity', title='Activities per FarmName in Last 7 Days')
    fig.update_layout(xaxis_title='FarmName', yaxis_title='Number of Activities')
    return fig

def create_dot_plot(df):
    df = df.astype(str)
    dot_data = []
    for column in df.columns:
        if column != 'Farm Name':
            for idx, value in enumerate(df[column]):
                color = None
                if value.startswith('well and passed'):
                    color = 'green'
                elif value.startswith('current'):
                    color = 'yellow'
                elif not value.startswith(('well and passed', 'current')):
                    color = 'red'
                if color:
                    dot_data.append({'Column': column, 'FarmName': df['Farm Name'][idx], 'Value': value, 'Color': color})
    dot_df = pd.DataFrame(dot_data)   
    dot_df = dot_df.sort_values(by='Column')   
    fig = px.scatter(dot_df, x='Column', y='FarmName', color='Color', 
                     color_discrete_map={'green': 'green', 'yellow': 'yellow', 'red': 'red'},
                     title="Growth Tracker Data",
                     labels={'FarmName': 'Farm Name'},
                     hover_data={'Value': True, 'Color': False})   
    fig.update_layout(height=1500)
    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))  
    return fig
    
def create_dot_plot_1(df):
    df = df.astype(str)
    dot_data = []
    columns_to_process = list(df.columns)
    if 'Farm Name' in columns_to_process:
        columns_to_process.remove('Farm Name')  # Remove 'Farm Name' from the list of columns to process  
    for column in columns_to_process:
        for idx, value in enumerate(df[column]):
            color = None
            if column == 'Sowing' or column == 'M0P/DAP' or column == 'UREA 1':
                try:
                    numeric_value = float(value.replace(',', '').split()[0])  # Remove commas and take the first part if it's a date
                    if (column == 'Sowing' and 7.5 <= numeric_value <= 8.5) or \
                       (column == 'M0P/DAP' and 9.5 <= numeric_value <= 10.5) or \
                       (column == 'UREA 1' and 6.5 <= numeric_value <= 7.5):
                        color = 'green'
                    else:
                        color = 'red'
                except ValueError:
                    pass  # Handle non-numeric or invalid format gracefully
            elif column == 'Weeding 1':
                if value.strip() and value.lower() not in ['nan', '0', 'null']:
                    color = 'green'
                else:
                    color = 'red'
            elif column == 'Irrigation 1':
                if value.strip() and value.lower() not in ['nan', '0', 'null']:
                    color = 'green'
                else:
                    color = 'red'
            if color:
                farm_name = df['Farm Name'][idx]
                farm_link = f"<a href='https://farmimage.streamlit.app/?farm_name={farm_name}' target='_blank'>{farm_name}</a>"
                dot_data.append({'Column': column, 'FarmName': farm_link, 'Value': value, 'Color': color})
    dot_df = pd.DataFrame(dot_data)
    
    # Ensure columns are sorted in the order they appear in the CSV sheet
    dot_df['Column'] = pd.Categorical(dot_df['Column'], categories=columns_to_process, ordered=True)
    dot_df = dot_df.sort_values(by='Column')

    fig = px.scatter(dot_df, x='Column', y='FarmName', color='Color',
                     color_discrete_map={'green': 'green', 'red': 'red'},
                     title="Growth Tracker Data",
                     category_orders={'Column': columns_to_process},  # Specify the category order for x-axis
                     labels={'FarmName': 'Farm Name'},
                     hover_data={'Value': True, 'FarmName': False, 'Color': False})  # Remove FarmName from hover data as it's now a link
    fig.update_layout(height=1500)
    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    return fig
    
def create_stacked_bar_chart(df):
    df = df.astype(str)
    status_counts = {}  
    for column in df.columns:
        if column != 'Farm Name':
            well_and_passed_count = df[df[column].str.startswith('well and passed', na=False)].shape[0]
            current_count = df[df[column].str.startswith('current', na=False)].shape[0]
            other_count = df[~df[column].str.startswith(('well and passed', 'current'), na=False)].shape[0]
            status_counts[column] = {
                'Well and Passed': well_and_passed_count,
                'Current': current_count,
                'Some Alert or not reached to this stage': other_count
            }   
    status_df = pd.DataFrame(status_counts).T.reset_index()
    status_df = status_df.melt(id_vars='index', var_name='Status', value_name='Count')    
    fig = px.bar(status_df, x='index', y='Count', color='Status', 
                 color_discrete_map={
                     'Well and Passed': 'green', 
                     'Current': 'yellow', 
                     'Some Alert or not reached to this stage': 'red'
                 },
                 title="Status Counts per Column in Growth Tracker",
                 labels={'index': 'Columns'})
    return fig
    
def kelvin_to_celsius(k):
    return k - 273.15

def plot_weather_trends(weather_df):
    weather_df['dt'] = pd.to_datetime(weather_df['dt'], errors='coerce')

    weather_df['temp_morn_C'] = kelvin_to_celsius(weather_df['temp_morn'])
    weather_df['temp_max_C'] = kelvin_to_celsius(weather_df['temp_max'])
    weather_df['temp_eve_C'] = kelvin_to_celsius(weather_df['temp_eve'])
    weather_df['temp_night_C'] = kelvin_to_celsius(weather_df['temp_night'])
    weather_df['temp_day_C'] = kelvin_to_celsius(weather_df['temp_day'])
    weather_df['temp_min_C'] = kelvin_to_celsius(weather_df['temp_min'])

    weather_df['wind_deg_normalized'] = weather_df['wind_deg'] / 360.0
    weather_df['clouds_normalized'] = weather_df['clouds'] / 100.0
    weather_df['humidity'] = weather_df['humidity'] / 100.0   
    weather_df['pressure'] = weather_df['pressure'] / 100.0

    fig = px.line(weather_df, x='dt', y=['temp_morn_C', 'temp_max_C', 'temp_eve_C', 'temp_night_C', 'temp_day_C', 'temp_min_C'],
                  title='Temperature Trends',
                  labels={'dt': 'Date', 'value': 'Temperature (°C)'},
                  line_shape='linear')

    #fig.add_scatter(x=weather_df['dt'], y=weather_df['pressure'], mode='markers', name='Pressure')
    fig.add_scatter(x=weather_df['dt'], y=weather_df['rain'], mode='markers', name='Rain')
    fig.add_scatter(x=weather_df['dt'], y=weather_df['snow'], mode='markers', name='Snow')
    fig.add_scatter(x=weather_df['dt'], y=weather_df['humidity'], mode='markers', name='Humidity')
    fig.add_scatter(x=weather_df['dt'], y=weather_df['clouds_normalized'], mode='markers', name='Clouds')
    fig.add_scatter(x=weather_df['dt'], y=weather_df['wind_speed'], mode='markers', name='Wind Speed')
    #fig.add_scatter(x=weather_df['dt'], y=weather_df['wind_deg_normalized'], mode='markers', name='Wind Degree')

    return fig
    
#activity_data_url = "https://raw.githubusercontent.com/sakshamraj4/abinbev/main/activity_avinbev.csv"
activity_data_url = "/home/az-134/Downloads/activity_avinbev.csv"
activity_df = pd.read_csv(activity_data_url)

new_data_url = "https://raw.githubusercontent.com/sakshamraj4/abinbev/main/data.csv"
new_data_df = pd.read_csv(new_data_url)

#growth_tracker_url = "https://raw.githubusercontent.com/sakshamraj4/abinbev/main/Growth_Tracker.csv"
growth_tracker_url = "/home/az-134/Downloads/Growth_Tracker.csv"
growth_tracker_df = pd.read_csv(growth_tracker_url)

growth_data_csv_url = "/home/az-134/Downloads/fig.csv"
growth_data_df = pd.read_csv(growth_data_csv_url)

weather_data_url = "/home/az-134/Downloads/weather_data.csv"
weather_df = pd.read_csv(weather_data_url)

menu_options = ['Organisation level Summary', 'Plot level Summary']
choice = st.sidebar.selectbox('Go to', menu_options)

if choice == 'Organisation level Summary':
    st.title("AB InBev Sustainability Dashboard")
    st.header("Organisation level Summarrization")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown('<div class="box"><h2>Total No of Plots</h2><p>59</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="box"><h2>Total Area (Bigha)</h2><p>82.28</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="box"><h2>Average Seed Rate</h2><p>8.01</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="box"><h2>Average DAP/MOP Rate</h2><p>10.02</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="box"><h2>Average Urea1 Rate</h2><p>7.01</p></div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="box"><h2>Average Urea2 Rate</h2><p>N/A</p></div>', unsafe_allow_html=True)
   
    create_bar_plots(activity_df)
    
    st.header("Growth Tracker Status")
    fig = create_stacked_bar_chart(growth_tracker_df)
    st.plotly_chart(fig)

elif choice == 'Plot level Summary':
    st.title("Plot level Summarization")
    st.header("Farm Activity in Last 7 Days")
    
    fig = create_line_chart(new_data_df)
    st.plotly_chart(fig)
    
    st.subheader("Weather Trends")
    fig_weather = plot_weather_trends(weather_df)
    st.plotly_chart(fig_weather)
    
    st.header("Growth Tracker Data")
    fig = create_dot_plot(growth_tracker_df)
    st.plotly_chart(fig)
    
    fig = create_dot_plot_1(growth_data_df)
    st.plotly_chart(fig)
    clicked_farm_name = st.experimental_get_query_params().get('farm_name', [None])[0]
    if clicked_farm_name:
        st.subheader(f"Detail view for Farm Name: {clicked_farm_name}")
        st.write(f"You clicked on {clicked_farm_name}. Add detailed view here.")
