import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
import requests, zipfile, io
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import json
import datetime as dt
st.title("USDA Economic Reseach Service Feed Grains Dashboard")
st.caption("This dashboard contains statistics on four feed grains (corn, grain sorghum, barley, and oats), foreign coarse grains (feed grains plus rye, millet, and mixed grains), hay, and related items. This includes data published in the monthly Feed Outlook and previously annual Feed Yearbook. Data are monthly, quarterly, and/or annual depending upon the data series. Latest data may be preliminary or projected. Missing values indicate unreported values, discontinued series, or not yet released data.")

zip_file_url = "https://www.ers.usda.gov/webdocs/DataFiles/50048/FeedGrains.zip?v=7455.2"

##function to format date
def date_format(freq, year, time, time_id):
  if freq == 1:
      return time +'-'+ year
  elif freq == 2:
    if str(time_id)[-1] == "1":
     return "Jan" +'-'+ year
    elif str(time_id)[-1] == "2":
      return "Apr" +'-'+ year
    elif str(time_id)[-1] == "3":
      return "July" +'-'+ year
    else:
      return "Oct" +'-'+ year
  elif freq == 3:
    return "Jan" +'-'+ year


#import data
@st.experimental_memo
def get_data():
    r = requests.get(zip_file_url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    df = pd.read_csv(z.open('FeedGrains.csv'))
    #format dates
    df['Year_ID'] = df['Year_ID'].apply(lambda x: str(x))
    df['Date'] = df.apply(lambda x: date_format(x["SC_Frequency_ID"], x["Year_ID"], x["Timeperiod_Desc"], x["Timeperiod_ID"]), axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    return df
df = get_data()


### sidebar
with st.sidebar:
    df = df[df.SC_GeographyIndented_Desc == "United States"]
    group = st.multiselect('Group', list(df.SC_Group_Desc.unique()), default='Prices')
    df_group = df[df.SC_Group_Desc.isin(group)]
  
    attr = st.multiselect('Data Attribute', list(df_group.SC_Attribute_Desc.unique()))
    attr_df = df_group[df_group.SC_Attribute_Desc.isin(attr)]
    comm = st.multiselect('Commodity', list(attr_df.SC_Commodity_Desc.unique()))
    comm_df= attr_df[attr_df.SC_Commodity_Desc.isin(comm)]
    dateFreq = st.radio('Frequency', list(comm_df.SC_Frequency_Desc.unique()))
    freq_df = comm_df[comm_df.SC_Frequency_Desc == dateFreq].rename(columns={"SC_Commodity_Desc":"Commodity"})

    comm_str = " "
    comm_delim = np.where(len(comm)>1,", ", " ")
    for c in enumerate(comm):
       comm_str =  comm_str + c[1] +  str(np.where(c[0]>-1,", ", " "))

#filters
unit = np.where(len(freq_df.SC_Unit_Desc.unique())<1,"", freq_df.SC_Unit_Desc.min())
if len(comm) <1:
    chart_title =  "<<< Select Data of Interest From the Sidebar To Get Started"
    st.subheader(chart_title)
    ###lottefile
    # GitHub: https://github.com/andfanilo/streamlit-lottie
    # Lottie Files: https://lottiefiles.com/

    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    

    #lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
    lottie_wheat = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_fogqgdsk.json")
   
    st_lottie(
        lottie_wheat,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
         # canvas
        height=None,
        width=None,
        key=None,
    )

else:
    chart_title =  "{}: {} {}".format(attr[0], comm_str , unit)
    st.subheader(chart_title)
    chart = alt.Chart(freq_df).mark_line().encode(
    x='Date',
    y= alt.Y('Amount', title = "{}".format(unit)),
    color='Commodity',
    strokeDash='Commodity',
    tooltip=['Commodity', 'Amount', 'SC_Unit_Desc', 'Date']).interactive()
    st.altair_chart(chart, use_container_width=True)
if len(comm) <1:
    table_title =  ""
else:
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(freq_df)

    st.download_button(
         label="Download The Data as CSV",
         data=csv,
         file_name='feed_grains.csv',
         mime='text/csv',
     )
    table_title = "Data Table: {}: {}".format(attr[0] , unit)
    st.text(table_title)
    displaytable =  freq_df[['Date', "Amount", "Commodity"]]
    displaytable = displaytable.set_index("Date")
    
    displaytable["Amount"] =  displaytable["Amount"].round(0)
   
    

    displaytable = pd.pivot_table(displaytable, values = 'Amount', index= 'Date', columns = 'Commodity')
    displaytable.index = displaytable.index.strftime("%b-%Y")

    st.dataframe(displaytable.style.set_precision(0))
##footer
st.write("Data Sources: [Feed Grains: Yearbook Table, USDA Economic Reseach Service](https://www.ers.usda.gov/data-products/feed-grains-database)")
st.text('By Sam Kobrin')


