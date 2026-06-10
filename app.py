import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Bengaluru Civic Efficiency", layout="wide")
st.title("Bengaluru Civic Efficiency Tracker")
st.markdown("Celebrating the municipal wards actively clearing their civic backlogs.")

@st.cache_data 
def load_data():
    df = pd.read_csv("bbmp_clean_2025.csv")
    # Clean out the unassigned messy data just like we did in SQL
    df = df[~df['Ward Name'].isin(['Non Ward', 'None'])]
    df = df.dropna(subset=['Ward Name'])
    return df

df = load_data()

total_complaints = len(df)
resolved_complaints = len(df[df['Grievance Status'] == 'Closed'])
overall_resolution = round((resolved_complaints / total_complaints) * 100, 1)

col1, col2, col3 = st.columns(3)
col1.metric("Total Complaints Logged", f"{total_complaints:,}")
col2.metric("Successfully Closed", f"{resolved_complaints:,}")
col3.metric("City-Wide Resolution Rate", f"{overall_resolution}%")

st.divider()

# 4. Calculate Resolution Rate per Ward (Replicating your SQL logic)
ward_stats = df.groupby('Ward Name').agg(
    Total_Complaints=('Grievance Status', 'count'),
    Closed_Issues=('Grievance Status', lambda x: (x == 'Closed').sum())
).reset_index()

ward_stats['Resolution Rate (%)'] = round((ward_stats['Closed_Issues'] / ward_stats['Total_Complaints']) * 100, 1)

top_wards = ward_stats[ward_stats['Total_Complaints'] > 100].sort_values(by='Resolution Rate (%)', ascending=False).head(10)

# 5. Build the Interactive Chart
st.subheader("The Top 10 Most Efficient Wards")
fig = px.bar(
    top_wards, 
    x='Ward Name', 
    y='Resolution Rate (%)',
    hover_data=['Total_Complaints', 'Closed_Issues'],
    color='Resolution Rate (%)',
    color_continuous_scale='Greens',
    title="Wards with the Highest Percentage of Resolved Complaints"
)
st.plotly_chart(fig, use_container_width=True)
