import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

from PIL import Image



def data():
    df = pd.read_csv("BostonCrime2022_8000_sample.csv")
    dd = pd.read_csv("BostonDistricts.csv")

    downtown = dd.iloc[0]['DISTRICT_NAME']
    charlestown = dd.iloc[1]['DISTRICT_NAME']
    east_boston = dd.iloc[2]['DISTRICT_NAME']
    roxbury = dd.iloc[3]['DISTRICT_NAME']
    mattapan = dd.iloc[4]['DISTRICT_NAME']
    south_boston = dd.iloc[5]['DISTRICT_NAME']
    dorchester = dd.iloc[6]['DISTRICT_NAME']
    south_end = dd.iloc[7]['DISTRICT_NAME']
    brighton = dd.iloc[8]['DISTRICT_NAME']
    west_roxbury = dd.iloc[9]['DISTRICT_NAME']
    jamaica_plain = dd.iloc[10]['DISTRICT_NAME']
    hyde_park = dd.iloc[11]['DISTRICT_NAME']

    data_new = df.copy()
    data_new.DISTRICT.replace(["A1"], downtown, inplace=True)
    data_new.DISTRICT.replace(["A15"], charlestown, inplace=True)
    data_new.DISTRICT.replace(["A7"], east_boston, inplace=True)
    data_new.DISTRICT.replace(["B2"], roxbury, inplace=True)
    data_new.DISTRICT.replace(["B3"], mattapan, inplace=True)
    data_new.DISTRICT.replace(["C6"], south_boston, inplace=True)
    data_new.DISTRICT.replace(["C11"], dorchester, inplace=True)
    data_new.DISTRICT.replace(["D4"], south_end, inplace=True)
    data_new.DISTRICT.replace(["D14"], brighton, inplace=True)
    data_new.DISTRICT.replace(["E5"], west_roxbury, inplace=True)
    data_new.DISTRICT.replace(["E13"], jamaica_plain, inplace=True)
    data_new.DISTRICT.replace(["E18"], hyde_park, inplace=True)
    data_new.drop(data_new.index[(data_new["DISTRICT"] == "External")], axis=0, inplace=True)
    data_new.drop(data_new.index[(data_new["Location"] == "(0,0)")], axis = 0, inplace = True)
    return data_new
#this function changes the district names from the code to the actual name
def time():
    df = data()
    df.sort_values(by='HOUR', inplace=True)
    return df
#sorts the times in order
def filter_data1(sel_district):
    df = data()
    df = df.loc[df['DISTRICT'].isin(sel_district)]

    return df


def count_districts(DISTRICTS, df):
    lst = [df.loc[df['DISTRICT'].isin([DISTRICT])].shape[0] for DISTRICT in DISTRICTS]
    return lst


def all_districts():
    df = data()
    lst = []
    for ind, row in df.iterrows():
        if row['DISTRICT'] not in lst:
            lst.append(row['DISTRICT'])

    return lst


def pie_chart(counts, sel_districts):
    plt.figure()

    explodes = [0 for i in range(len(counts))]
    maximum = counts.index(np.max(counts))
    explodes[maximum] = 0.25

    plt.pie(counts, labels=sel_districts, explode=explodes, autopct="%.2f")
    return plt


def user_inputs1():
    st.sidebar.header("Pie Chart Inputs")
    district = st.sidebar.multiselect("Select District(s): ", all_districts())
    st.sidebar.write("You selected the districts:", district)
    button = st.sidebar.button("Make Chart")

    return (district, button)
#inputs for pie chart

def filter_data2(weekdays):
    df = data()
    df = df.loc[df['DAY_OF_WEEK'].isin(weekdays)]

    return df


def count_weekdays(days, df):
    lst = [df.loc[df['DAY_OF_WEEK'].isin([day])].shape[0] for day in days]
    return lst


def all_days():
    df = data()
    lst = []
    for ind, row in df.iterrows():
        if row['DAY_OF_WEEK'] not in lst:
            lst.append(row['DAY_OF_WEEK'])
    return lst


def bar_plot(count, sel_weekdays):
    df = data()
    fig = px.bar(df, x=sel_weekdays, y=count)
    return fig


def user_inputs2():
    st.sidebar.header("Bar Chart Inputs")
    days = st.sidebar.multiselect("Choose Weekday(s): ", all_days())
    st.sidebar.write("You selected the days:", days)
    button = st.sidebar.button("Make Chart")
    return (days, button)

#inputs for bar chart


def filter_data3(df,sel_hour):
    df = df.loc[df['HOUR'].isin(sel_hour)]

    return df


def count_time(hours, df):
    lst = [df.loc[df['HOUR'].isin([hour])].shape[0] for hour in hours]
    return lst


def all_time():
    df = time()
    lst = []
    for ind, row in df.iterrows():
        if row['HOUR'] not in lst:
            lst.append(row['HOUR'])
    return lst

def hours(df):
    hour = [row['HOUR'] for ind,row in df.iterrows()]
    districts = [row['DISTRICT'] for ind,row in df.iterrows()]

    dict = {}
    for district in districts:
        dict[district] = []

    for i in range(len(hour)):
        dict[districts[i]].append(hour[i])

    return dict

def average_hour(dict_hour):
    dict = {}
    for key in dict_hour.keys():
        dict[key] = np.mean(dict_hour[key])

    return dict

def line_chart(hour,average):

    df = time()
    average = average_hour(hours(df))
    x = average.keys()
    y = average.values()

    fig,ax = plt.subplots(figsize = (15,5))
    ax.plot(x,y,linewidth=2.0)




    ax.set_xlabel('Hours')
    ax.set_ylabel('Average Hour')
    ax.set_title('Average Crime per Hour based on District')

    ax.legend(title = "District", loc = "upper left")


    return plt

def user_inputs3():
    st.sidebar.header("Line Chart Inputs")
    start_hour, end_hour = st.sidebar.select_slider("Select a Range of Hours: ", options= all_time(),value=(1,5))
    button = st.sidebar.button("Make Chart")

    return(start_hour,end_hour,button)

#inputs for line plot

def user_inputs4():
    st.sidebar.header("Map Inputs")
    district = st.sidebar.multiselect("Select District(s): ", all_districts())
    button = st.sidebar.button("Make Map")

    return (district, button)
#inputs for map
def gen_map(sel_district):
    df = filter_data1(sel_district)
    map_df= df.filter(['OFFENSE_DESCRIPTION','DISTRICT','Lat','Long'])

    view_state = pdk.ViewState(latitude=map_df['Lat'].mean(), longitude=map_df['Long'].mean(), zoom=100)
    layer = pdk.Layer('ScatterplotLayer',data=map_df,get_position='[Long, Lat]',get_color =[100,175,250],get_radius=30, pickable=True)
    tool_tip = {'html': 'Offense:<br/> <b>{OFFENSE_DESCRIPTION}</b>, {DISTRICT}','style':{'backgroundColor':'steelblue', 'color': 'white'}}
    map = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
                   initial_view_state= view_state,
                   layers=[layer],
                   tooltip= tool_tip)
    st.pydeck_chart(map)


page = st.sidebar.radio("Choose your page", ["Welcome", "Pie Chart", "Bar Chart", "Line Chart","Map"])


if page == "Welcome":
    st.title("Analysis of Boston Crime Data")
    name = st.sidebar.text_input("Enter your name:", "")
    img = Image.open("Boston-MAP.webp")
    if name:
        st.header(f'Welcome {name}')
        st.subheader("Use the buttons to navigate through different pages")
        st.image(img, width=500)
        st.write("""Throughout this webpage, you will be able to interact with a data 
        set relating to crime in the Greater Boston Area.""")


elif page == "Pie Chart":
    st.header("Crime by District")
    district, button = user_inputs1()
    data1 = filter_data1(district)
    series = count_districts(district, data1)

    if len(district) > 0 and button:
        st.pyplot(pie_chart(series, district))
        st.dataframe(series)

elif page == "Bar Chart":
    st.header("Crime by Weekday")
    days, button = user_inputs2()

    data2 = filter_data2(days)
    series = count_weekdays(days, data2)

    if len(days) > 0 and button:
        st.plotly_chart(bar_plot(series, days))
        st.dataframe(series)

elif page == "Line Chart":
    st.header("Line Chart")
    start_hour,end_hour,button = user_inputs3()
    hour = range(start_hour,end_hour+1)
    dat = time()
    data3 = filter_data3(dat,hour)
    average = average_hour(hours(data3))
    if button:
        st.pyplot(line_chart(hour,average))

elif page == "Map":
    st.header("Crime by District")
    district, button = user_inputs4()
    data1 = filter_data1(district)

    if len(district) > 0 and button:
        gen_map(district)

