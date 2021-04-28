import streamlit as st
import numpy as np
from numpy import cos, sin, arcsin, sqrt
import pandas as pd
import matplotlib as plt
from matplotlib import Patch
import seaborn as sns
import statistics as stats
import pydeck as pdk


def read_file(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna('Unknown')          # Replaces 'NaN'
    df = df.replace('M�xico and Central America', 'México and Central America')         # Allows python to read 'é' in the file
    lst = []
    columns = ['Volcano Number', 'Volcano Name', 'Country', 'Primary Volcano Type', 'Activity Evidence', 'Last Known Eruption', 'Region', 'Subregion', 'Latitude', 'Longitude', 'Elevation (m)', 'Dominant Rock Type', 'Tectonic Setting']
    for index, row in df.iterrows():
        sub = []
        for col in columns:
            index_no = df.columns.get_loc(col)
            sub.append(row[index_no])
        lst.append(sub)

    return lst


def distance_calculator(data, volc1, volc2):
    CHECK = True
    volc1_list = []
    volc2_list = []
    while CHECK:
        for sublist in data:
            if sublist[1] == volc1:
                volc1_list.append(sublist)          # Finds the sublist containing volcano 1 and isolates it
                CHECK = False
    CHECK = True
    while CHECK:
        for sublist in data:
            if sublist[1] == volc2:
                volc2_list.append(sublist)
                CHECK = False

    volc1_lat = volc1_list[0][8]
    volc1_lon = volc1_list[0][9]                        # finds latitude and longitude using indexes
    volc2_lat = volc2_list[0][8]
    volc2_lon = volc2_list[0][9]

    # See Reference 1
    # Converts latitude and longitude angles to radians

    RADIUS = 3956               # radius of earth in miles
    KM = 1.609344

    lat1_rad = np.radians(volc1_lat)
    lon1_rad = np.radians(volc1_lon)
    lat2_rad = np.radians(volc2_lat)
    lon2_rad = np.radians(volc2_lon)
    lat_dist = lat2_rad - lat1_rad
    lon_dist = lon2_rad - lon1_rad

    dist1 = sin(lat_dist / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(lon_dist / 2)**2
    distance_miles = 2 * arcsin(sqrt(dist1)) * RADIUS
    distance_km = distance_miles * KM

    st.write(f'The distance between {volc1} and {volc2} is {distance_miles: .2f} miles ({distance_km: .2f} kilometers)')


def volc_map(data, volc1, volc2):
    CHECK = True
    volc1_list = []
    volc2_list = []
    while CHECK:
        for sublist in data:
            if sublist[1] == volc1:
                volc1_list.append(sublist)
                CHECK = False
    CHECK = True
    while CHECK:
        for sublist in data:
            if sublist[1] == volc2:
                volc2_list.append(sublist)
                CHECK = False

    volc1_lat = volc1_list[0][8]
    volc1_lon = volc1_list[0][9]
    volc2_lat = volc2_list[0][8]
    volc2_lon = volc2_list[0][9]

    coord_list = [(volc1_lon, volc1_lat), (volc2_lon, volc2_lat)]
    # coord_array2 = np.array([[volc1, volc1_lat, volc1_lon], [volc2, volc2_lat, volc2_lon]])      # Creates an array with lat and lon
    # coord_df2 = pd.DataFrame(data=coord_array, columns=['Volcano', 'lat', 'lon'])

    coord_array = np.array([[volc1_lat, volc1_lon], [volc2_lat, volc2_lon]])
    coord_df = pd.DataFrame(coord_array, columns=['lat', 'lon'])

    st.subheader('Volcano Locations')
    st.map(data=coord_df)


# Reads file and returns data as a dataframe


def panda_frame(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna('Unknown')
    df = df.replace('M�xico and Central America', 'México and Central America')
    del df['Link']
    del df['Volcano Number']

    return df


def frequency_create(data, attribute, bar_colors='OrRd'):

    if attribute == 'Region':
        k = 6
    elif attribute == 'Dominant Rock Type':
        k = 11
    elif attribute == 'Tectonic Setting':
        k = 12
    att_list = []
    freq_dict = {}                      # Uses a dictionary to track frequency
    for sublist in data:
        att_list.append(sublist[k])
    for word in att_list:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1

    data_tuples = [(value, key) for value, key in freq_dict.items()]
    df = pd.DataFrame(data_tuples, columns=['Attribute', 'Frequency'])

    colors = sns.color_palette(bar_colors, n_colors=len(df))                            # Color pallets for bars
    fig, ax = plt.subplots()
    sns.barplot(x=df.index, y='Frequency', data=df, palette=colors).set_title(f'Frequency of {attribute}s', size=20)    # plots index on x axis
    plt.grid(axis='y', color='grey')
    fig.set_size_inches(16, 8)
    leg_map = dict(zip(df.Attribute, colors))                                           # Creates a custom legend, matches color to x value
    entries = [Patch(color=v, label=k) for k, v in leg_map.items()]
    plt.legend(handles=entries, loc='upper right')
    st.pyplot(fig)


def attribute_stats(data, attribute):
    stats_list = []
    if attribute == 'Latitude':                                 # Statistics
        k = 8
    elif attribute == 'Longitude':
        k = 9
    elif attribute == 'Elevation':
        k = 10
    for sublist in data:
        stats_list.append(sublist[k])

    data_mean = stats.mean(stats_list)
    data_median = stats.median(stats_list)
    data_mode = stats.mode(stats_list)
    data_std = stats.stdev(stats_list)
    data_max = max(stats_list)
    data_min = min(stats_list)

    if attribute == 'Elevation':
        st.write(f'The mean is {data_mean: .2f} meters  \n'
                 f'The median is {data_median: .2f} meters  \n'
                 f'The mode is {data_mode: .2f} meters  \n'
                 f'The standard deviation is {data_std: .2f} meters     \n'
                 f'The maximum value is {data_max: .2f} meters  \n'
                 f'The minimum value is {data_min: .2f} meters  \n')
    else:
        st.write(f'The mean is {data_mean: .2f} degrees     \n'
                 f'The median is {data_median: .2f} degrees     \n'
                 f'The mode is {data_mode: .2f} degrees     \n'
                 f'The standard deviation is {data_std: .2f} degrees    \n'
                 f'The maximum value is {data_max: .2f} degrees     \n'
                 f'The minimum value is {data_min: .2f} degrees     \n')


def main():
    data = read_file('volcanoes.csv')
    st.title('Volcano Web Application')
    st.subheader('Welcome, please select an option from the sidebar.')
    st.sidebar.subheader('Command Sidebar')

    stat_choice = st.sidebar.radio('Select Data to Analyze', ('Hide Statistics', 'Latitude', 'Longitude', 'Elevation'))
    if stat_choice != 'Hide Statistics':                                        # Added functionality to hide graphs
        st.subheader(f'{stat_choice} Statistics')
        attribute_stats(data, stat_choice)

    freq_choice = st.sidebar.selectbox('Select Attribute to Create Frequency Plot', ('Hide Chart', 'Region', 'Dominant Rock Type', 'Tectonic Setting'))

    if freq_choice != 'Hide Chart':
        st.subheader(f'{freq_choice} Frequency Chart')
        frequency_create(data, freq_choice)
        bar_select = st.selectbox('Select a new color pallet', ('Orange to Red', 'Blue Fade', 'Pastels', 'Green to Blue', 'Peach to Purple', 'Rainbow'))
        if bar_select == 'Orange to Red':
            frequency_create(data, freq_choice, 'OrRd')                 # Allows user to choose color other than the default the function lists
        if bar_select == 'Blue Fade':
            frequency_create(data, freq_choice, 'Blues')
        elif bar_select == 'Pastels':
            frequency_create(data, freq_choice, 'husl')
        elif bar_select == 'Green to Blue':
            frequency_create(data, freq_choice, 'crest')
        elif bar_select == 'Peach to Purple':
            frequency_create(data, freq_choice, 'flare')
        elif bar_select == 'Rainbow':
            frequency_create(data, freq_choice, 'Spectral')

    volcano_map_choice = st.sidebar.radio('Show Map and Distance Between Two Volcanoes', ('Hide', 'Show'))

    if volcano_map_choice != 'Hide':
        volc1 = st.sidebar.text_input("First Volcano Name")
        volc2 = st.sidebar.text_input("Second Volcano Name")
    else:
        volc1 = ''
        volc2 = ''

    if volc1 and volc2:
        distance_calculator(data, volc1, volc2)             # uses both the distance and map functions
        volc_map(data, volc1, volc2)

    dataframe_show = st.sidebar.radio('Dataframe:', ('Hide', 'Show'))

    if dataframe_show != 'Hide':
        df = panda_frame('volcanoes.csv')
        filter_list = []                                        # Dataframe sorting and filtering
        st.subheader('Sort Data')
        reset_button = st.button('Press to Reset:')
        if reset_button:
            dataframe_show = 'Hide'
            dataframe_show = 'Show'
            colfilter_button = ''
            elevfilter_button = ''
            filter_list = []
        name_sort = st.radio('Volcano Name:', ('Ascending', 'Descending'))
        elev_filter = st.slider('Elevation Cutoff:', -5700, 6879)
        col_list = ('Volcano Name', 'Country', 'Primary Volcano Type', 'Activity Evidence',
                    'Last Known Eruption',	'Region', 'Subregion', 'Latitude', 'Longitude',
                    'Elevation (m)', 'Dominant Rock Type', 'Tectonic Setting')
        column_filter = st.multiselect('Select Columns to Filter', col_list)

        if name_sort == 'Ascending':
            df = df.sort_values('Volcano Name', ascending=True)
            st.dataframe(df)

        if name_sort == 'Descending':
            df = df.sort_values('Volcano Name', ascending=False)
            st.dataframe(df)

        if elev_filter:
            elevfilter_button = st.button('Press to Filter by Elevation:')
            if elevfilter_button:
                df = df[df['Elevation (m)'] >= elev_filter]
                st.dataframe(df)

        if len(column_filter) > 0:
            colfilter_button = st.button('Press to Filter by Column:')
            if colfilter_button:
                for i in column_filter:                                 # Creates a list based on multiselect, # checks to column names
                    filter_list.append(i)
                for x in col_list:
                    if x not in column_filter:
                        del df[x]
                st.dataframe(df)


main()



# References
# 1)
# https://www.igismap.com/haversine-formula-calculate-geographic-distance-earth/
#
# I used this source to learn about the Haversine formula. This formula calculates
# the distance between coordinates using latitude longitude.
# 2)
# https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
# I had some difficulty graphing my x axis labels for my frequency plots. This was because of
# the length of labels. I found a workaround by using seaborn and using the legend to display my x axis tickers
