import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import logging
import plotly.figure_factory as ff

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up Streamlit page configuration
st.set_page_config(layout="wide")

# Set up password protection
PASSWORD = "Abinbev@123"

def password_protection():
    def check_password():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            st.session_state["password_attempts"] = 0
        else:
            st.session_state["password_correct"] = False
            st.session_state["password_attempts"] += 1

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_attempts"] = 0

    if not st.session_state["password_correct"]:
        if st.session_state["password_attempts"] >= 3:
            st.error("Too many incorrect attempts. Please refresh the page to try again.")
        else:
            st.text_input("Enter password:", type="password", on_change=check_password, key="password")
            if st.session_state["password_correct"]:
                st.success("Password correct! Access granted.")
            elif st.session_state["password_attempts"] > 0:
                st.error("Password incorrect. Please try again.")
        st.stop()

password_protection()

st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #333;
    }
    .stApp {
        background-color: #ffffff;
    }
    .box-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .box {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        width: 150px;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .box h2 {
        color: #333;
        margin-bottom: 10px;
        font-size: 1em;
    }
    .box p {
        font-size: 1.5em;
        color: #555;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def create_bar_plots(df):
    for column in df.columns:
        if column not in ['Tillage Operation', 'Farm Name']:
            st.subheader(f"{column}")
            total_count = df.shape[0]
            done_count = df[df[column].isin(['Done', 'Done early', 'Done on time'])].shape[0]
            df_plot = pd.DataFrame({
                'Status': ['Total', 'Done and Pop Followed'],
                'Count': [total_count, done_count]
            })
            colors = ['blue', 'green']
            fig = px.bar(df_plot, x='Count', y='Status', orientation='h', text='Count', width=800, height=200, color='Status', color_discrete_sequence=colors)
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                xaxis_title="Count",
                yaxis_title="Status",
                title=f"{column}",
                showlegend=False
            )
            st.plotly_chart(fig)
            
def create_activity_progress_plot():
    data = [
        dict(Task='Sowing', Start='2024-05-15', Finish='2024-05-20', Done='59', NotDone='0'),
        dict(Task='DAP / MOP Fertilizer', Start='2024-05-15', Finish='2024-05-20', Done='59', NotDone='0'),
        dict(Task='Weeding 1', Start='2024-06-05', Finish='2024-06-11', Done='59', NotDone='0'),
        dict(Task='Irrigation 1', Start='2024-06-13', Finish='2024-06-15', Done='59', NotDone='0'),
        dict(Task='Urea 1', Start='2024-06-15', Finish='2024-06-17', Done='59', NotDone='0'),
        dict(Task='Weeding 2', Start='2024-06-28', Finish='2024-07-01', Done='6', NotDone='53')
    ]

    data_combined = []
    for item in data:
        data_combined.append(dict(Task=item['Task'], Start=item['Start'], Finish=item['Finish'], Resource='Done: ' + item['Done'], Done=item['Done']))
        data_combined.append(dict(Task=item['Task'], Start=item['Start'], Finish=item['Finish'], Resource='Not Done: ' + item['NotDone'], Done=item['NotDone']))

    fig = ff.create_gantt(data_combined, index_col='Resource', group_tasks=True, showgrid_x=True, showgrid_y=True)

    for bar in fig['data']:
        if 'Done' in bar['name']:
            bar['marker']['color'] = 'green'
            bar['hoverinfo'] = 'text+name'
            bar['text'] = [item['Done'] for item in data_combined if item['Resource'] == bar['name']]
            bar['textposition'] = 'middle center'
        elif 'Not Done' in bar['name']:
            bar['marker']['color'] = 'red'
            bar['hoverinfo'] = 'text+name'
            bar['text'] = [item['Done'] for item in data_combined if item['Resource'] == bar['name']]
            bar['textposition'] = 'middle center'

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Activity"
    )

    st.plotly_chart(fig)

def create_line_chart(df, time_frame):
    try:
        logging.info("Converting 'Date' column to datetime format.")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        if df['Date'].isnull().any():
            logging.warning("There are NaT values in 'Date' column after conversion.")
        
        if time_frame == 'Last 2 Days':
            last_days = datetime.now() - timedelta(days=2)
            df_filtered = df[df['Date'] >= last_days]
        elif time_frame == 'Last Week':
            last_days = datetime.now() - timedelta(days=7)
            df_filtered = df[df['Date'] >= last_days]
        elif time_frame == 'Last Month':
            last_days = datetime.now() - timedelta(days=30)
            df_filtered = df[df['Date'] >= last_days]
        elif time_frame == 'Visit Till Date':
            df_filtered = df[df['Date'] <= datetime.now()]
        else:
            raise ValueError("Invalid time_frame value provided.")
        
        if df_filtered.empty:
            logging.warning("The filtered dataframe is empty.")
        
        logging.info("Grouping by 'FarmName' and 'Date' to count activities.")
        df_counts = df_filtered.groupby(['FarmName', 'Date']).size().reset_index(name='Activity')
        
        logging.info("Grouping by 'FarmName' to count unique dates with activities.")
        df_counts = df_counts.groupby('FarmName').size().reset_index(name='Activity')
        
        logging.info("Creating the line chart.")
        fig = px.line(df_counts, x='FarmName', y='Activity', title=f'Activities per Plot in {time_frame}')
        fig.update_layout(xaxis_title='FarmName', yaxis_title='Number of Activities')
        
        logging.info("Line chart created successfully.")
        return fig
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

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
                     color_discrete_map={
                         'green': 'green',
                         'yellow': 'yellow',
                         'red': 'red'
                     },
                     title="Plot wise Growth Stage Summary",
                     labels={'FarmName': 'Farm Name'},
                     hover_data={'Value': True, 'Color': False})
    fig.for_each_trace(lambda t: t.update(name = {'green': 'well and passed', 
                                                  'yellow': 'current stage', 
                                                  'red': 'not in current stage or some Alert'}[t.name]))

    fig.update_layout(height=1500)
    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    return fig

def create_dot_plot_1(df):
    df = df.astype(str)
    dot_data = []
    columns_to_process = list(df.columns)
    if 'Farm Name' in columns_to_process:
        columns_to_process.remove('Farm Name')
    for column in columns_to_process:
        for idx, value in enumerate(df[column]):
            color = None
            if column in ['Sowing', 'M0P/DAP', 'UREA 1']:
                try:
                    numeric_value = float(value.replace(',', '').split()[0])
                    if (column == 'Sowing' and 7.5 <= numeric_value <= 8.5) or \
                       (column == 'M0P/DAP' and 9.5 <= numeric_value <= 10.5) or \
                       (column == 'UREA 1' and 6.5 <= numeric_value <= 7.5):
                        color = 'green'
                    else:
                        color = 'red'
                except ValueError:
                    pass
            elif column in ['Weeding 1', 'Irrigation 1', 'Weeding 2']:
                if value.strip() and value.lower() not in ['nan', '0', 'null']:
                    color = 'green'
                else:
                    color = 'red'
            if color:
                farm_name = df['Farm Name'][idx]
                farm_link = f"<a href='https://farmimage.streamlit.app/?farm_name={farm_name}' target='_blank'>{farm_name}</a>"
                dot_data.append({'Column': column, 'FarmName': farm_link, 'Value': value, 'Color': color})
    
    dot_df = pd.DataFrame(dot_data)
    dot_df['Column'] = pd.Categorical(dot_df['Column'], categories=columns_to_process, ordered=True)
    dot_df = dot_df.sort_values(by='Column')
    
    fig = px.scatter(dot_df, x='Column', y='FarmName', color='Color',
                     color_discrete_map={'green': 'green', 'red': 'red'},
                     title="Plot wise Activity Summary",
                     category_orders={'Column': columns_to_process},
                     labels={'FarmName': 'Farm Name'},
                     hover_data={'Value': True, 'FarmName': False, 'Color': False})
    
    # Update the trace names for the legend
    fig.for_each_trace(lambda t: t.update(name={
        'green': 'well and pop Followed', 
        'red': 'pop not followed or Activity not Done'
    }[t.name]))

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
                 title="Growth Tracker Summary",
                 labels={'index': 'Columns'})
    return fig

def kelvin_to_celsius(k):
    return k - 273.15

def plot_weather_trends(weather_df):
    weather_df['dt'] = pd.to_datetime(weather_df['dt'], errors='coerce')
    weather_df['temp_max_C'] = kelvin_to_celsius(weather_df['temp_max'])
    weather_df['temp_min_C'] = kelvin_to_celsius(weather_df['temp_min'])
    weather_df['clouds_normalized'] = weather_df['clouds'] / 100.0
    weather_df['snow_emoji'] = weather_df['snow'].apply(lambda x: '❄️' if x > 0 else '')
    weather_df['rain_emoji'] = weather_df['rain'].apply(lambda x: '🌧️' if x > 0 else '')
    weather_df['clouds_emoji'] = weather_df['clouds_normalized'].apply(lambda x: '☁️' if x > 0.5 else '')
    fig = px.line(weather_df, x='dt', y=['temp_max_C', 'temp_min_C'],
                  title='Temperature Trends',
                  labels={'dt': 'Date', 'value': 'Temperature (°C)'},
                  line_shape='linear')
    fig.add_scatter(x=weather_df['dt'], y=[weather_df['temp_max_C'].max()] * len(weather_df),
                    mode='text', text=weather_df['snow_emoji'], name='Snow', textposition='top center')
    fig.add_scatter(x=weather_df['dt'], y=[weather_df['temp_min_C'].max()] * len(weather_df),
                    mode='text', text=weather_df['rain_emoji'], name='Rain', textposition='top center')
    fig.add_scatter(x=weather_df['dt'], y=[weather_df['temp_min_C'].min()] * len(weather_df),
                    mode='text', text=weather_df['clouds_emoji'], name='Clouds', textposition='top center')
    fig.update_layout(height=700)
    return fig

# Load data
activity_data_url = "https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main/activity_avinbev.csv"
activity_df = pd.read_csv(activity_data_url)
new_data_url = "https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main/data.csv"
new_data_df = pd.read_csv(new_data_url)
growth_tracker_url = "https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main/Growth_Tracker.csv"
growth_tracker_df = pd.read_csv(growth_tracker_url)
growth_data_csv_url = "https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main/fig.csv"
growth_data_df = pd.read_csv(growth_data_csv_url)
weather_data_url = "https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main//weather_data.csv"
weather_df = pd.read_csv(weather_data_url)
activity_progress_url = 'https://github.com/sakshamraj4/Abinbav_sustainability/raw/main/crop_monitorig_protocol.csv'
activity_progress_df = pd.read_csv(activity_progress_url)
field_team_url = 'https://raw.githubusercontent.com/sakshamraj4/Abinbav_sustainability/main/field_team_data.csv'
field_team_df = pd.read_csv(field_team_url)

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
        
    
    st.title("Crop Monitoring Observation")
    create_activity_progress_plot()
    
    st.header("Activity Progress")
    create_bar_plots(activity_df)
    
    st.header("Growth Tracker Status")
    fig = create_stacked_bar_chart(growth_tracker_df)
    st.plotly_chart(fig)
    
    st.download_button(
        label="Download Data updated via Field team",
        data=new_data_df.to_csv(index=False).encode('utf-8'),
        file_name='Daily_visit_data.csv',
        mime='text/csv'
    )
    
    st.download_button(
        label="Download Data Entered via app",
        data=field_team_df.to_csv(index=False).encode('utf-8'),
        file_name='Field_team_updated_data.csv',
        mime='text/csv'
    )

elif choice == 'Plot level Summary':
    st.title("Plot level Summarization")
    st.header("Plot Visit Summary by Field Team")

    time_frame_options = ['Last Week', 'Last Month', 'Visit Till Date']
    selected_time_frame = st.selectbox('Select Time Frame', time_frame_options)
    
    fig = create_line_chart(new_data_df, selected_time_frame)
    st.plotly_chart(fig)
    
    st.download_button(
        label="Download Daily visit Data",
        data=new_data_df.to_csv(index=False).encode('utf-8'),
        file_name='Daily_visit_data.csv',
        mime='text/csv'
    )
    
    st.subheader("Weather Trends")
    fig_weather = plot_weather_trends(weather_df)
    st.plotly_chart(fig_weather)
    
    st.download_button(
        label="Download Weather Data",
        data=weather_df.to_csv(index=False).encode('utf-8'),
        file_name='Weather_data.csv',
        mime='text/csv'
    )
    
    st.header("Growth Tracker Data")
    fig = create_dot_plot(growth_tracker_df)
    st.plotly_chart(fig)
    
    st.download_button(
        label="Download Growth Stage Data",
        data=growth_tracker_df.to_csv(index=False).encode('utf-8'),
        file_name='growth_tracker_data.csv',
        mime='text/csv'
    )
    
    fig = create_dot_plot_1(growth_data_df)
    st.plotly_chart(fig)
    clicked_farm_name = st.experimental_get_query_params().get('farm_name', [None])[0]
    if clicked_farm_name:
        st.subheader(f"Detail view for Farm Name: {clicked_farm_name}")
        st.write(f"You clicked on {clicked_farm_name}. Add detailed view here.")
        
    st.download_button(
        label="Download Activity Status Data",
        data=growth_data_df.to_csv(index=False).encode('utf-8'),
        file_name='activity_tracker_data.csv',
        mime='text/csv'
    )
