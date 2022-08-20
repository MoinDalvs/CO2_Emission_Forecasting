import streamlit as st
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pickle import load
from io import BytesIO
import pickle
import requests
import calendar
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import plot as off
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# load the model from disk
#yearly_model = load(open('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_yearly.pkl?raw=true', 'rb'))

mfile = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_yearly.pkl?raw=true').content)
yearly_model = pickle.load(mfile)

#monthly_model = load(open('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_monthly.pkl?raw=true', 'rb'))

mfile1 = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_monthly.pkl?raw=true').content)
monthly_model = pickle.load(mfile1)

data = pd.read_excel("https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/Data%20set/CO2%20dataset.xlsx?raw=true",index_col='Year',parse_dates=True)
data.reset_index(inplace=True)
data["Year"] = pd.to_datetime(data.Year,format="%b-%y")

raw_data=pd.read_excel('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/Data%20set/CO2%20dataset.xlsx?raw=true'\
                   ,index_col='Year',parse_dates=True)

resampled_data=pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Interpolated_CO2_dataset.csv'\
                , index_col='Year',parse_dates=True)

visual = pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Visual_CO2.csv')
visual_1 = pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Visual_CO2_1.csv')

st.title("Forecasting $CO_2$ Emission")
st.sidebar.subheader("Select one of the 'Option' below")
nav = st.sidebar.radio("",["Time Series Analysis","Yearly Forecasting", "Monthly Forecasting"])

if nav == "Time Series Analysis":
    st.subheader(f"Select the Time Series from the drop down list")
    dropdown = st.selectbox('', ['None','Raw Data "Yearly"','Resampled Data "Monthly"'])
    

    if dropdown == 'Raw Data "Yearly"':

       st.write("Raw Data")
       raw_data

       st.subheader("Area plot of the data") 
       st.area_chart(data=raw_data.CO2, width=150, height=300, use_container_width=True)
   

       st.subheader("Line plot of the data") 
       st.line_chart(data=raw_data.CO2, width=150, height=300, use_container_width=True)
       
       st.subheader("Lowest and Highest $CO_2$ Emission over the Years")
       st.set_option('deprecation.showPyplotGlobalUse', False)
       data.plot(kind ='line',x = 'Year', y = 'CO2', figsize = (16,8), marker = 'o')

       plt.title('Amount of CO2 Emission over the Years', fontdict={'fontsize': 16,'fontweight':'bold'})
       plt.ylabel('CO2 Emission', fontdict={'fontsize': 14})
       plt.xlabel('Year', fontdict={'fontsize': 14})

       minimum = data[data.CO2==data.CO2.min()]
       maximum = data[data.CO2==data.CO2.max()]
       minimum.reset_index(inplace=True, drop=True)
       minimum = minimum.iat[0,0]
       maximum.reset_index(inplace=True, drop=True)
       maximum = maximum.iat[0,0]

       label_list = [(minimum, 'On 1845\nLowest\n$CO_2$ Emission\nat 0.00175', 'g'),(maximum, "On 1979\nHighest\n$CO_2$ Emission\nat 18.2", 'r')]

       ax = plt.gca()

       for date_point, label, clr in label_list:
          plt.axvline(x=date_point, color=clr)
          plt.text(date_point, ax.get_ylim()[1]-10, label,
             horizontalalignment='center',
             verticalalignment='center',
             color=clr,
             bbox=dict(facecolor='white', alpha=0.9),fontsize=12)
       plt.tight_layout()
       st.pyplot()

       st.subheader("Histogram of the data") 
       fig= plt.figure(figsize=(10,4))
       plt.hist(data.CO2)
       st.pyplot(fig)

       st.subheader("Time Series Decomposition") 

       seasonality=pd.read_excel('C:/Users/Moin Dalvi/Documents/Data Science Material/Projects/Air Quality Forecasting/CO2 dataset.xlsx'\
                   ,index_col='Year',parse_dates=True)
       st.text('Additive Seasonality')
       decompose_ts_add = seasonal_decompose(seasonality.CO2, model='additive')
       with plt.rc_context():
          plt.rc("figure", figsize=(14,10))
          decompose_ts_add.plot()
          st.pyplot()

       st.text('Multiplicative Seasonality')
       decompose_ts_mul = seasonal_decompose(seasonality.CO2, model='multiplicative')
       with plt.rc_context():
          plt.rc("figure", figsize=(14,10))
          decompose_ts_add.plot()
          st.pyplot()
    
      
       
    if dropdown == 'Resampled Data "Monthly"':

       st.write("Resampled Data")
       resampled_data

       st.subheader("Line plot of the data") 
       st.line_chart(data=resampled_data.CO2, width=150, height=300, use_container_width=True)

       century = visual_1.groupby('Century').agg({"CO2" : "mean"}).reset_index() 

       plot_century = century[['CO2']].plot(kind = 'bar', figsize=(12,8),color = ('red'))
       plt.xlabel('Centuries',fontsize=14)
       plt.ylabel('CO2 Emission Rate',fontsize=14)
       plt.xticks(np.arange(3), ('$18^{th}$\nCentury', '$19^{th}$\nCentury', '$20^{th}$\nCentury'),rotation = 'horizontal',fontsize=13)
       plt.title('Average CO2 Emission Throughout Centuries',fontsize=16, fontweight='bold')
       # label the bar
       for rec, label in zip(plot_century.patches,
                            century['CO2'].round(2).astype(str)):
          plot_century.text(rec.get_x() + rec.get_width()/2, 
                            rec.get_height() + 0.1, 
                            label+'',  
                            ha = 'center', 
                            color = 'black',
                           fontsize='x-large'
                           )
       plt.tight_layout()
       st.pyplot()

       decade = visual_1.groupby('Decade').min().reset_index()

       plot_decade = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('green'))
       plt.xlabel('Decades',fontsize=14)
       plt.ylabel('CO2 Emission Rate',fontsize=14)
       plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
       plt.title('Lowest CO2 Emission Throughout Decades',fontsize=16, fontweight='bold')
       # label the bar
       for rec, label in zip(plot_decade.patches,
                            decade['CO2'].round(2).astype(str)):
          plot_decade.text(rec.get_x() + rec.get_width()/2, 
                            rec.get_height() + 0.1, 
                            label+'',  
                            ha = 'center', 
                            color = 'black',
                           fontsize='large')
       plt.tight_layout()
       st.pyplot()
       
       decade = visual_1.groupby('Decade').max().reset_index()
 
       plot_decade = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('red'))
       plt.xlabel('Decades',fontsize=14)
       plt.ylabel('CO2 Emission Rate',fontsize=14)
       plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
       plt.title('Highest CO2 Emission Throughout Decades',fontsize=16, fontweight='bold')
       # label the bar
       for rec, label in zip(plot_decade.patches,
                            decade['CO2'].round(2).astype(str)):
          plot_decade.text(rec.get_x() + rec.get_width()/2, 
                            rec.get_height() + 0.1, 
                            label+'',  
                            ha = 'center', 
                            color = 'black',
                           fontsize='large')
       plt.tight_layout()
       st.pyplot()

       decade = visual_1.groupby('Decade').agg({"CO2" : "mean"}).reset_index() 

       plot_decade = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('blue'))
       plt.xlabel('Decades',fontsize=14)
       plt.ylabel('CO2 Emission Rate',fontsize=14)
       plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
       plt.title('Average CO2 Emission Throughout Decade',fontsize=16, fontweight='bold')
       # label the bar
       for rec, label in zip(plot_decade.patches,
                            decade['CO2'].round(2).astype(str)):
          plot_decade.text(rec.get_x() + rec.get_width()/2, 
                            rec.get_height() + 0.1, 
                            label+'',  
                            ha = 'center', 
                            color = 'black',
                           fontsize='large')
       plt.tight_layout()
       st.pyplot()

       st.write("Click below to plot an Average Monthly, Quarterly and Weekly $CO_2$ Emission Analysis")
       if st.button("Monthly Quarterly and Weekly Analysis"):

          df_m_sa = visual.groupby('month').agg({"CO2" : "mean"}).reset_index()
          df_m_sa['CO2'] = round(df_m_sa['CO2'],2)
          df_m_sa['month_text'] = df_m_sa['month'].apply(lambda x: calendar.month_abbr[x])
          df_m_sa['text'] = df_m_sa['month_text'] + ' - ' + df_m_sa['CO2'].astype(str) 

          df_w_sa = visual.groupby('week').agg({"CO2" : "mean"}).reset_index() 
          df_q_sa = visual.groupby('quarter').agg({"CO2" : "mean"}).reset_index() 
          # chart color
          df_m_sa['color'] = '#496595'
          df_m_sa['color'][:-1] = '#c6ccd8'
          df_w_sa['color'] = '#c6ccd8'

          # chart
          fig = make_subplots(rows=2, cols=2, vertical_spacing=0.08,
                             row_heights=[0.7, 0.3], 
                             specs=[[{"type": "bar"}, {"type": "pie"}],
                                    [{"colspan": 2}, None]],
                             column_widths=[0.7, 0.3],
                             subplot_titles=("Month wise Average", "Quarter wise Average", 
                                             "Week wise Average"))

          fig.add_trace(go.Bar(x=df_m_sa['CO2'], y=df_m_sa['month'], marker=dict(color= df_m_sa['color']),
                              text=df_m_sa['text'],textposition='auto',
                              name='Month', orientation='h'), 
                              row=1, col=1)
          fig.add_trace(go.Pie(values=df_q_sa['CO2'], labels=df_q_sa['quarter'], name='Quarter',
                              marker=dict(colors=['#334668','#496595','#6D83AA','#91A2BF','#C8D0DF']), hole=0.7,
                              hoverinfo='label+percent+value', textinfo='label+percent'), 
                              row=1, col=2)
          fig.add_trace(go.Scatter(x=df_w_sa['week'], y=df_w_sa['CO2'], mode='lines+markers', fill='tozeroy', fillcolor='#c6ccd8',
                              marker=dict(color= '#496595'), name='Week'), 
                              row=2, col=1)

          # styling
          fig.update_yaxes(visible=False, row=1, col=1)
          fig.update_xaxes(visible=False, row=1, col=1)
          fig.update_xaxes(tickmode = 'array', tickvals=df_w_sa.week, ticktext=[i for i in df_w_sa.week], 
                          row=2, col=1)
          fig.update_yaxes(visible=False, row=2, col=1)
          fig.update_layout(height=750, bargap=0.15,
                           margin=dict(b=0,r=20,l=20), 
                           title_text="Average CO2 Emission Analysis",
                           template="plotly_white",
                           title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                           font=dict(color='#8a8d93'),
                           hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                           showlegend=False)
          fig.show()

         
       st.write("Click below to plot an Average Daily $CO_2$ Emission Analysis")
       if st.button("Daily Analysis"):
          df_dw_sa = visual.groupby('day_of_week').agg({"CO2" : "mean"}).reset_index()
          df_dw_sa.CO2 = round(df_dw_sa.CO2, 2)

          # chart
          fig = px.bar(df_dw_sa, y='day_of_week', x='CO2', title='Avg CO2 Emission vs Day of Week',
                      color_discrete_sequence=['#c6ccd8'], text='CO2',
                      category_orders=dict(day_of_week=["Monday","Tuesday","Wednesday","Thursday", "Friday","Saturday","Sunday"]))
          fig.update_yaxes(showgrid=False, ticksuffix=' ', showline=False)
          fig.update_xaxes(visible=False)
          fig.update_layout(margin=dict(t=60, b=0, l=0, r=0), height=350,
                           hovermode="y unified", 
                           yaxis_title=" ", template='plotly_white',
                           title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                           font=dict(color='#8a8d93'),
                           hoverlabel=dict(bgcolor="#c6ccd8", font_size=13, font_family="Lato, sans-serif"))
          fig.show()





if nav == "Yearly Forecasting":
    
    st.sidebar.subheader("Select the Year to Forecast from 2015")
    years = st.sidebar.slider("",2015,2034,step = 1)

    lists = int(years)
    if lists == 2015:
       year=1
    elif lists == 2016:
      year=2
    elif lists == 2017:
       year=3
    elif lists == 2018:
       year=4
    elif lists == 2019:
       year=5
    elif lists == 2020:
       year=6
    elif lists == 2021:
       year = 7
    elif lists == 2022:
       year = 8
    elif lists == 2023:
       year = 9
    elif lists == 2024:
       year = 10
    elif lists == 2025:
       year = 11
    elif lists == 2026:
       year = 12
    elif lists == 2027:
       year = 13
    elif lists == 2028:
       year = 14
    elif lists == 2029:
       year = 15
    elif lists == 2030:
       year = 16
    elif lists == 2031:
       year = 17
    elif lists == 2032:
       year = 18
    elif lists == 2033:
       year = 19
    elif lists == 2034:
       year = 20

    st.sidebar.subheader("To Forecast till the Selected Years\n Please Click on the 'Forecast' Button")
    
    
    pred_yearly = pd.DataFrame()
    pred_yearly['CO2 Emission'] = yearly_model.forecast(year)
      

    if st.sidebar.button("Forecast"):
       st.sidebar.subheader(f"Forecasted for the selected Year")
       st.sidebar.write(pred_yearly[-1:])
       st.write(f"$CO_2$ Emission Forecasted from year 2015" )
       pred_yearly
       fig = plt.figure(figsize=(12,4), dpi=100)
       plt.plot(pred_yearly['CO2 Emission'], label='Auto regression forecast (ARIMA)')
       plt.title('Forecasted for next {} years from 2015'.format(year))
       plt.legend(loc='best')

       st.write("Line plot for the Forecasted data")
       st.pyplot(fig)

if nav == "Monthly Forecasting":
    
    st.sidebar.subheader("Select the Month to Forecast from 2014 February")
    drop = st.sidebar.selectbox('', ['Feb 2014','Mar 2014','Apr 2014','May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014','Dec 2014',
                                     'Jan 2015','Feb 2015','Mar 2015','Apr 2015','May 2015','Jun 2015','Jul 2015','Aug 2015','Sep 2015','Oct 2015','Nov 2015','Dec 2015',
                                     'Jan 2016','Feb 2016','Mar 2016','Apr 2016','May 2016','Jun 2016','Jul 2016','Aug 2016','Sep 2016','Oct 2016','Nov 2016','Dec 2016',
                                     'Jan 2017','Feb 2017','Mar 2017','Apr 2017','May 2017','Jun 2017','Jul 2017','Aug 2017','Sep 2017','Oct 2017','Nov 2017','Dec 2017',
                                     'Jan 2018','Feb 2018','Mar 2018','Apr 2018','May 2018','Jun 2018','Jul 2018','Aug 2018','Sep 2018','Oct 2018','Nov 2018','Dec 2018',
                                     'Jan 2019','Feb 2019','Mar 2019','Apr 2019','May 2019','Jun 2019','Jul 2019','Aug 2019','Sep 2019','Oct 2019','Nov 2019','Dec 2019',
                                     'Jan 2020','Feb 2020','Mar 2020','Apr 2020','May 2020','Jun 2020','Jul 2020','Aug 2020','Sep 2020','Oct 2020','Nov 2020','Dec 2020',
                                     'Jan 2021','Feb 2021','Mar 2021','Apr 2021','May 2021','Jun 2021','Jul 2021','Aug 2021','Sep 2021','Oct 2021','Nov 2021','Dec 2021',
                                     'Jan 2022','Feb 2022','Mar 2022','Apr 2022','May 2022','Jun 2022','Jul 2022','Aug 2022','Sep 2022','Oct 2022','Nov 2022','Dec 2022',
                                     'Jan 2023'])
    
    if drop == 'Feb 2014':
       months=1
    elif drop == 'Mar 2014':
      months=2
    elif drop == 'Apr 2014':
       months=3
    elif drop == 'May 2014':
       months=4
    elif drop == 'Jun 2014':
       months=5
    elif drop == 'July 2014':
       months=6
    elif drop == 'Aug 2014':
       months = 7
    elif drop == 'Sep 2014':
       months = 8
    elif drop == 'Oct 2014':
       months = 9
    elif drop == 'Nov 2014':
       months = 10
    elif drop == 'Dec 2014':
       months = 11
    elif drop == 'Jan 2015':
       months = 12
    elif drop == 'Feb 2015':
       months=13
    elif drop == 'Mar 2015':
      months=14
    elif drop == 'Apr 2015':
       months=15
    elif drop == 'May 2015':
       months=16
    elif drop == 'Jun 2015':
       months= 17
    elif drop == 'July 2015':
       months=18
    elif drop == 'Aug 2015':
       months = 19
    elif drop == 'Sep 2015':
       months = 20
    elif drop == 'Oct 2015':
       months = 21
    elif drop == 'Nov 2015':
       months = 22
    elif drop == 'Dec 2015':
       months = 23
    elif drop == 'Jan 2016':
       months = 24
    elif drop == 'Feb 2016':
       months= 25
    elif drop == 'Mar 2016':
      months= 26
    elif drop == 'Apr 2016':
       months= 27
    elif drop == 'May 2016':
       months=28 
    elif drop == 'Jun 2016':
       months=29
    elif drop == 'July 2016':
       months=30
    elif drop == 'Aug 2016':
       months = 31
    elif drop == 'Sep 2016':
       months = 32
    elif drop == 'Oct 2016':
       months = 33
    elif drop == 'Nov 2016':
       months = 34
    elif drop == 'Dec 2016':
       months = 35
    elif drop == 'Jan 2017':
       months = 36
    elif drop == 'Feb 2017':
       months= 37
    elif drop == 'Mar 2017':
      months= 38
    elif drop == 'Apr 2017':
       months= 39
    elif drop == 'May 2017':
       months= 40
    elif drop == 'Jun 2017':
       months= 41
    elif drop == 'July 2017':
       months= 42
    elif drop == 'Aug 2017':
       months = 43
    elif drop == 'Sep 2017':
       months = 44
    elif drop == 'Oct 2017':
       months = 45
    elif drop == 'Nov 2017':
       months = 46
    elif drop == 'Dec 2017':
       months = 47
    elif drop == 'Jan 2018':
       months = 48
    elif drop == 'Feb 2018':
       months= 49
    elif drop == 'Mar 2018':
      months= 50
    elif drop == 'Apr 2018':
       months= 51
    elif drop == 'May 2018':
       months=52
    elif drop == 'Jun 2018':
       months=53
    elif drop == 'July 2018':
       months=54
    elif drop == 'Aug 2018':
       months = 55
    elif drop == 'Sep 2018':
       months = 56
    elif drop == 'Oct 2018':
       months = 57
    elif drop == 'Nov 2018':
       months = 58
    elif drop == 'Dec 2018':
       months = 59
    elif drop == 'Jan 2019':
       months = 60
    elif drop == 'Feb 2019':
       months=61
    elif drop == 'Mar 2019':
      months=62
    elif drop == 'Apr 2019':
       months=63
    elif drop == 'May 2019':
       months=64
    elif drop == 'Jun 2019':
       months=65
    elif drop == 'July 2019':
       months=66
    elif drop == 'Aug 2019':
       months = 67
    elif drop == 'Sep 2019':
       months = 68
    elif drop == 'Oct 2019':
       months = 69
    elif drop == 'Nov 2019':
       months = 70
    elif drop == 'Dec 2019':
       months = 71
    elif drop == 'Jan 2020':
       months = 72
    elif drop == 'Feb 2020':
       months=73
    elif drop == 'Mar 2020':
      months=74
    elif drop == 'Apr 2020':
       months=75
    elif drop == 'May 2020':
       months=76
    elif drop == 'Jun 2020':
       months=77
    elif drop == 'July 2020':
       months=78
    elif drop == 'Aug 2020':
       months = 79
    elif drop == 'Sep 2020':
       months = 80
    elif drop == 'Oct 2020':
       months = 81
    elif drop == 'Nov 2020':
       months = 82
    elif drop == 'Dec 2020':
       months = 83
    elif drop == 'Jan 2021':
       months = 84
    elif drop == 'Feb 2021':
       months= 85
    elif drop == 'Mar 2021':
      months= 86
    elif drop == 'Apr 2021':
       months= 87
    elif drop == 'May 2021':
       months= 88
    elif drop == 'Jun 2021':
       months= 89
    elif drop == 'July 2021':
       months= 90
    elif drop == 'Aug 2021':
       months = 91
    elif drop == 'Sep 2021':
       months = 92
    elif drop == 'Oct 2021':
       months = 93
    elif drop == 'Nov 2021':
       months = 94
    elif drop == 'Dec 2021':
       months = 95
    elif drop == 'Jan 2022':
       months = 96
    elif drop == 'Feb 2022':
       months= 97
    elif drop == 'Mar 2022':
      months= 98
    elif drop == 'Apr 2022':
       months= 99
    elif drop == 'May 2022':
       months= 100
    elif drop == 'Jun 2022':
       months= 101
    elif drop == 'July 2022':
       months= 102
    elif drop == 'Aug 2022':
       months = 103
    elif drop == 'Sep 2022':
       months = 104
    elif drop == 'Oct 2022':
       months = 105
    elif drop == 'Nov 2022':
       months = 106
    elif drop == 'Dec 2022':
       months = 107
    elif drop == 'Jan 2023':
       months = 108
    

    st.sidebar.subheader("To Forecast till the Selected Months\n Please Click on the 'Forecast' Button")
    
    
    pred_monthly = pd.DataFrame()
    pred_monthly['CO2 Emission'] = monthly_model.forecast(months)
    

    if st.sidebar.button("Forecast"):
       st.sidebar.write(f"Forecasted for the selected Month")
       st.sidebar.write(pred_monthly[-1:])
       st.write(f"$CO_2$ Emission Forecasted from Feb 2014" )
       pred_monthly
       fig1 = plt.figure(figsize=(12,4), dpi=100)
       plt.plot(pred_monthly['CO2 Emission'], label='Auto regression forecast (ARIMA)')
       plt.title('Forecasted for next {} months from 2014 Feb'.format(months))
       plt.legend(loc='best')

       st.write("Line plot for the Forecasted data")
       st.pyplot(fig1)

