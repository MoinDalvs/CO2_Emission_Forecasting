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
import requests
import calendar
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import plot as off
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
st.set_option('deprecation.showPyplotGlobalUse', False)
import time

# load the model from the disk
mfile = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_yearly.pkl?raw=true').content)
yearly_model = load(mfile)

mfile1 = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/CO2_Forecast_arima_monthly.pkl?raw=true').content)
monthly_model = load(mfile1)

mfile2 = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/yearly_train_arima.pkl?raw=true').content)
yearly_model_test = load(mfile2)

mfile13 = BytesIO(requests.get('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/monthly_train_arima.pkl?raw=true').content)
monthly_model_test = load(mfile13)

# load the data from the disk

data = pd.read_excel("https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/Data%20set/CO2%20dataset.xlsx?raw=true",index_col='Year',parse_dates=True)
data.reset_index(inplace=True)
data["Year"] = pd.to_datetime(data.Year,format="%b-%y")

raw_data=pd.read_excel('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/Data%20set/CO2%20dataset.xlsx?raw=true'\
                   ,index_col='Year',parse_dates=True)

resampled_data=pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Interpolated_CO2_dataset.csv'\
                , index_col='Year',parse_dates=True)

visual = pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Visual_CO2.csv')
visual_1 = pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Visual_CO2_1.csv')


st.sidebar.subheader("Select an Option below")
nav = st.sidebar.radio("",['About',"Yearly Forecasting", "Monthly Forecasting","Time Series Analysis","Model Evaluation"])

if nav == 'About':

    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.02)
       my_bar.progress(percent_complete + 1)
    
    st.sidebar.markdown('#### Project By\n\
<table>\
<tr>\
<td>\
     <img src="https://avatars.githubusercontent.com/u/99672298?v=4" width="180"/>\
     \
     moindalvs@gmail.com\
\
<p align="center">\
<a href = "https://github.com/MoinDalvs"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>\
<a href = "https://twitter.com/DalvsHubot"><img src = "https://www.shareicon.net/download/2016/07/06/107115_media.svg" width="36" height="36"/></a>\
<a href = "https://www.linkedin.com/in/moin-dalvi-277b0214a//"><img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/></a>\
</p>\
</td>\
</tr> \
  </table>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💾 About Project","🦸‍♂️ About Me"])
    
    with tab1:

       st.header(f"Business Objective: To Forecast CO2 levels for an organization so that the organization can follow government norms with respects to CO2 emission levels.")

       st.write(f"Note: Select other options from the left sidebar")

       st.image("https://geographical.co.uk/wp-content/uploads/carbon-dioxide-emissions-title-1200x800.jpg")

       st.markdown('### **Introduction:**\n\
#### There is wide consensus among scientists and policymakers that global warming as defined by the Intergovernmental Panel on Climate Change (IPCC) should be pegged at 1.5 Celsius above the pre-industrial level of warming in order to maintain environmental sustainability . The threats and risks of climate change have been evident in the form of various extreme climate events, such as tsunamis, glacier melting, rising sea levels, and heating up of the atmospheric temperature. Emissions of greenhouse gases, such as carbon dioxide (CO2) are the main cause of global warming. \n\
\
#### In this Project for Time Series Analysis and Forecasting\n\
\
#### Each step taken like EDA, Feature Engineering, Model Building,  Model Evaluation and Prediction table, and Deployment. Explaining which model to select on basis of metrics like RMSE, MAPE and MAE value for each model. Finally explaining which model we will use for Forecasting.\n\
\
#### Better accuracy in short-term forecasting is required for intermediate planning for the target to reduce CO2 emissions. High stake climate change conventions need accurate predictions of the future emission growth path of the participating organization to make informed decisions.\n\
\
#### Exponential techniques, Linear statistical modeling and Autoregressive models are used to forecast the emissions and the best model will be selected on these basis\n\
\
##### 1.) Minimum error\n\
##### 2.) Low bias and low variance trade off')

    with tab2:
      
       st.markdown('<img align="center" alt="code"  height="180" width="720" src = "https://user-images.githubusercontent.com/99672298/186088252-ef41cefc-cda7-49e8-960d-efbd59d3696c.jpg">\
\
<h1 align="center">Hi 👋, I"m Dalvi Moin</h1>\n\
<h3 align="center">A passionate Data Science Enthusiast</h3>\n\
\n\
- 🌱 I love learning as a process and mostly about new things...\n\
\n\
- 🔭 I’m currently working as an Intern at <a href="https://aivariant.com/" target="blank"><img align="center" src="https://www.aivariant.com/wp-content/uploads/2021/01/ai_logo5_2-1.png" alt="MoinDalvs" height="25" width="70" /></a>\n\
\n\
- 🌱 I’m currently learning **Machine Learning and Deep Learning**\n\
\n\
- 💞️ I’m looking to collaborate on all topics related to  Data Science, Machine Learning and Artificial Intellegence.\n\
\n\
- 👀 I’m interested in Data Science, Machine Learning and Artificial Intelligence\n\
\n\
- 🌱 I’m currently mastering Python, Tableau, R, MySQL, Azure, Apache Spark, Hadoop, SAS, Artificial intellegence and Deep learning\n\
\n\
- 📫 You can reach me on my email id moindalvs@gmail.com\n\
\n\
<img align="right" alt="code"  height="400" width="400" src = "https://user-images.githubusercontent.com/94888819/179503858-d2f6d197-7a3f-495b-888c-5a60679bed94.gif"> \n\
\
<img align="left" src="https://user-images.githubusercontent.com/94888819/179538709-781ca826-4b36-42e7-aeda-ad6b07e719ea.png" alt="pp" width="50" height="50" /> </a>\
<img align="left" alt="SQL" height="40" src="https://raw.githubusercontent.com/habc0d3r/0th-project/master/icons8-sql-96.png" /> <b>\
<img align="left" alt="excel" height="40" src="https://d3j0t7vrtr92dk.cloudfront.net/stembakuniversity/1616519913_png-clipart-microsoft-excel-computer-icons-microsoft-template-angle-removebg-preview.png" />\
<img src="https://user-images.githubusercontent.com/94888819/179532814-fa9beb8f-0fd6-4160-8d47-650af59c58a1.png" alt="GC" width="40" height="40"/> </a> \
<img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> <a href="https://unity.com/" target="_blank" rel="noreferrer"> </p> \
<img src="https://user-images.githubusercontent.com/94888819/179531328-610ccc8c-11cc-40cf-82dc-5902d473b7e1.png" alt="tab" width="40" height="40"/> </a> \
<img align="left" alt="teminal" height="50" \
src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/terminal/terminal.png" /> </a>\
<a href="https://code.visualstudio.com" target="_blank" rel="noreferrer"> <img src="https://user-images.githubusercontent.com/94888819/179420219-9ce785fd-b1eb-4caa-aebd-086c088d05f2.png" alt="vsc" width="60" height="40"/> </a>\
<a href="https://github.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/habc0d3r/0th-project/master/icons8-github.svg" alt="github" width="50" height="50"/> </a> \
<a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a>\
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> </p>\
<img align="left" alt="pycharm" height="40" src="https://user-images.githubusercontent.com/94888819/179536543-d438fb65-8501-475d-9f8b-5bbc18bab86a.png" />\
<img align="left" alt="jn" height="50" src="https://user-images.githubusercontent.com/94888819/179537708-2241ab23-8c86-40c6-8fa9-f3979be75ade.png" /> \
<img src="https://user-images.githubusercontent.com/94888819/179422108-0108ecc4-96d4-4a9f-93d2-f4f2ea98688e.png" alt="stream" width="40" height="50"/> </a>\
<a href="https://aws.amazon.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" alt="aws" width="40" height="40"/> </a>\
<a href="https://azure.microsoft.com/en-in/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/microsoft_azure/microsoft_azure-icon.svg" alt="azure" width="40" height="40"/> </a>\
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="50" height="50"/> </a> </p>\
\
Connect with me:\
<p align="left">\
<a href="https://www.linkedin.com/in/moin-dalvi-277b0214a" target="blank"><img align="center" src="https://img.icons8.com/color/344/linkedin-circled--v1.png" alt="MoinDalvs" height="50" width="50" /></a>\
<a href="https://www.kaggle.com/moindalvi" target="blank"><img align="center" src="https://img.icons8.com/external-tal-revivo-color-tal-revivo/344/external-kaggle-an-online-community-of-data-scientists-and-machine-learners-owned-by-google-logo-color-tal-revivo.png" alt="MoinDalvs" height="40" width="40" /></a>\
<a href="https://www.youtube.com/channel/UCN7xdRsVxEaaH5H0f3eNNmg" target="blank"><img align="center" src="https://img.icons8.com/color/344/youtube-play.png" alt="MoinDalvs" height="50" width="50" /></a>\
</p>\
\
<img align="center" alt="code"  height="300" width="600" src = "https://static.wixstatic.com/media/3e99b9_f53a1cab95ae4dfd938a1bf6a1a62f49~mv2.gif">\n\
\n\
\
Github Status: \n\
<p align="center"> <img width="50%" src="https://github-readme-stats.vercel.app/api?username=MoinDalvs&show_icons=true&theme=tokyonight"/> </p>\
<p align="center"> <img src="http://github-readme-streak-stats.herokuapp.com?user=MoinDalvs&theme=neon-palenight" width="50%"  alt="streak"> </p>\
<p align="center"> <a href="https://github.com/ryo-ma/github-profile-trophy"><img src="https://github-profile-trophy.vercel.app/?username=MoinDalvs&show_icons=true&theme=discord" alt="MoinDalvs" />\
</a> </p>\
\
<div style="align:center;\
            display:fill;\
            border-radius: false;\
            border-style: solid;\
            border-color:#000000;\
            border-style: false;\
            border-width: 2px;\
            color:#CF673A;\
            font-size:15px;\
            font-family: Georgia;\
            background-color:#E8DCCC;\
            text-align:center;\
            letter-spacing:0.1px;\
            padding: 0.1em;">\
\
**<h2>♡ Thank you for taking the time ♡**', unsafe_allow_html=True)

if nav == "Time Series Analysis":

    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.08)
       my_bar.progress(percent_complete + 1)
    
    st.title("Time Series Analysis")
    st.subheader(f"Select the Time Series from the drop down list")
    dropdown = st.selectbox('', ['Raw Data "Yearly"','Resampled Data "Monthly"'])
    

    if dropdown == 'Raw Data "Yearly"':
        
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗃 Raw Data","📈 Line Plot", "📈 Yearly Insights", "📈 Centuries Insights","📈 Decade Insights"])
       
        with tab1:

            tab1.write("Raw Data")
            tab1.write(raw_data)


        with tab2:

            st.subheader("Line plot of the data") 
            st.line_chart(data=raw_data.CO2, width=150, height=300, use_container_width=True)
        with tab3:
       
            st.subheader("Lowest and Highest $CO_2$ Emission over the Years")
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
        with tab4:
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

        with tab5:

            col1, col2, col3 = st.columns(3)
            with col1:

                decade = visual_1.groupby('Decade').min().reset_index()

                plot_decade_min = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('green'))
                plt.xlabel('Decades',fontsize=14)
                plt.ylabel('CO2 Emission Rate',fontsize=14)
                plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
                plt.title('Lowest CO2 Emission Throughout Decades',fontsize=16, fontweight='bold')
                # label the bar
                for rec, label in zip(plot_decade_min.patches,
                                    decade['CO2'].round(2).astype(str)):
                  plot_decade_min.text(rec.get_x() + rec.get_width()/2, 
                                    rec.get_height() + 0.1, 
                                    label+'',  
                                    ha = 'center', 
                                    color = 'black',
                                   fontsize='large')
                plt.tight_layout()
                st.pyplot()
            
            with col2:

                decade = visual_1.groupby('Decade').max().reset_index()
         
                plot_decade_max = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('red'))
                plt.xlabel('Decades',fontsize=14)
                plt.ylabel('CO2 Emission Rate',fontsize=14)
                plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
                plt.title('Highest CO2 Emission Throughout Decades',fontsize=16, fontweight='bold')
                # label the bar
                for rec, label in zip(plot_decade_max.patches,
                                    decade['CO2'].round(2).astype(str)):
                  plot_decade_max.text(rec.get_x() + rec.get_width()/2, 
                                    rec.get_height() + 0.1, 
                                    label+'',  
                                    ha = 'center', 
                                    color = 'black',
                                   fontsize='large')
                plt.tight_layout()
                st.pyplot()
            
            with col3:

                decade = visual_1.groupby('Decade').agg({"CO2" : "mean"}).reset_index() 

                plot_decade_avg = decade[['CO2']].plot(kind = 'bar', figsize=(14,10),color = ('blue'))
                plt.xlabel('Decades',fontsize=14)
                plt.ylabel('CO2 Emission Rate',fontsize=14)
                plt.xticks(np.arange(21), labels=(decade.Decade),rotation = 'horizontal',fontsize=13)
                plt.title('Average CO2 Emission Throughout Decade',fontsize=16, fontweight='bold')
                # label the bar
                for rec, label in zip(plot_decade_avg.patches,
                                    decade['CO2'].round(2).astype(str)):
                  plot_decade_avg.text(rec.get_x() + rec.get_width()/2, 
                                    rec.get_height() + 0.1, 
                                    label+'',  
                                    ha = 'center', 
                                    color = 'black',
                                   fontsize='large')
                plt.tight_layout()
                st.pyplot()
             
    if dropdown == 'Resampled Data "Monthly"':
        tab1, tab2, tab3, tab4 = st.tabs(["🗃 Resampled Data","📈 Line Plot", "📈 Day of Week Insights", "📈 Seasonal Insights"])
        with tab1:
            st.write("Resampled Data")
            resampled_data

        with tab2:

            st.subheader("Line plot of the data") 
            st.line_chart(data=resampled_data.CO2, width=150, height=300, use_container_width=True)

        with tab3:

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
            st.plotly_chart(fig)

        with tab4:

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
            st.plotly_chart(fig)

if nav == "Model Evaluation":
    
    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.1)
       my_bar.progress(percent_complete + 1)
    

    st.title("Model Evaluation")
    st.subheader(f"Select the Model")
    radios = st.radio("",["Yearly Model","Monthly Model"])
    
    if radios == "Yearly Model":
        
        st.write(f"Yearly Model Evaluation")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Accuracy","📈 Train Test Split","📈 Predicted Data", "❌ Error","⛅ Predicted vs Actual Data"])

        model_data=pd.read_excel('https://github.com/MoinDalvs/CO2_Emission_Forecasting/blob/main/Data%20set/CO2%20dataset.xlsx?raw=true'\
                   ,index_col='Year',parse_dates=True)
        model_data.index = pd.DatetimeIndex(model_data.index.values,freq=model_data.index.inferred_freq)
        model_data.head()
        size = -20
        train_data = model_data[:size]
        test_data = model_data[size:]

        # ARIMA Model
        y_hat_ar = model_data.copy()
        pred = yearly_model_test.forecast(20)
        y_hat_ar['arima_forecast'] = yearly_model_test.predict(model_data.index.min(), model_data.index.max())
        
        with tab2:
            st.write(f'Splitting Train and Test Data on Raw data. Leaving Test Data with 20 Years of Time Series. We are going to forecast for the last 20 years, that is from 1994 to 2014.')

            train_data.CO2.plot(legend=True,label='TRAIN',color='green')
            test_data.CO2.plot(legend=True,label='TEST', figsize=(18,6),color='red')
            plt.xlabel('Year', fontsize= 12)
            plt.legend()
            st.pyplot()

            plt.figure(figsize=(16,4))
            test_data.CO2.plot(label="org")
            plt.title("Test data Series", fontsize=14)
            st.pyplot()
            plt.figure(figsize=(16,4))
            test_data["CO2"].rolling(5).mean().plot(label=str(5))
            plt.title("Moving Average "+str(5), fontsize=14)
            plt.legend(loc='best')
            st.pyplot()

        with tab3:

            # Testing vs forecasted 
            plt.figure(figsize=(12,6), dpi=200)
            plt.plot(test_data, label='Test')
            plt.plot(pred, label='Auto regression forecast (ARIMA)')
            plt.legend(loc='best')
            st.pyplot()

        with tab5:
            #Calculating Absolute Percent Error and Error
            # Computing the absolute percent error
            col1, col2 = st.columns(2)
            with col1:
                APE=100*(abs(y_hat_ar['CO2']-y_hat_ar['arima_forecast'])/y_hat_ar['CO2'])
                Error = (abs(y_hat_ar['CO2']-y_hat_ar['arima_forecast']))

                # adding absolute percent error to table
                y_hat_ar['Error']=Error
                y_hat_ar['Absolute Percent Error %']=APE

                y_hat_ar
            with col2:
                #Visualizing the Relationship between the Actual and Predicted ValuesModel Validation
                plt.figure(figsize=(12,8))
                plt.xlabel("Actual Values", fontsize =12)
                plt.ylabel("Predicted values", fontsize =12)
                plt.title("The Scatterplot of Relationship between Actual Values and Predictions", fontsize =16, fontweight = 'bold')
                plt.scatter(y_hat_ar['CO2'], y_hat_ar['arima_forecast'])
                st.pyplot()

        with tab4:
            # Error Evaluation
            yearly_model_test.plot_diagnostics(figsize=(16,8))
            st.pyplot()

        with tab1:

            # Accuracy vs Error Rate

            plt.figure(figsize=(16,8)) 
            # Setting size in Chart based on 
            # given values
            Error_Rate = np.mean(APE[size:])
            Accuracy = 100-np.mean(APE[size:])
            sizes = [Accuracy, Error_Rate]
              
            # Setting labels for items in Chart
            labels = ['Accuracy', 'Error']
              
            # colors
            colors = ['#007500', '#FF0000']
              
            # explosion
            explode = (0.0, 0.05)
              
            # Pie Chart
            plt.pie(sizes, colors=colors, labels=labels,
                    autopct='%1.1f%%', shadow=True,
                    pctdistance=0.85, 
                    explode=explode,
                    startangle=0.0,
                    textprops = {'size':'x-large',
                               'fontweight':'bold',
                                'rotation':'0.0',
                               'color':'black'})
              
            # draw circle
            centre_circle = plt.Circle((0.0, 0.0), 0.70, fc='white')
            fig = plt.gcf()
              
            # Adding Circle in Pie chart
            fig.gca().add_artist(centre_circle)

            # Adding Title of chart
            plt.title('ARIMA Model \n Accuracy and Error Rate \n on Test Data', fontsize = 16, fontweight = 'bold')
              
            # Add Legends
            plt.legend(labels, loc="upper right")
              
            # Displaying Chart
            st.pyplot()

    if radios == "Monthly Model":
        
        st.write(f"Yearly Model Evaluation")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Accuracy","📈 Train Test Split","📈 Predicted Data", "❌ Error","⛅ Predicted vs Actual Data"])

        model_data=pd.read_csv('https://raw.githubusercontent.com/MoinDalvs/CO2_Emission_Forecasting/main/Data%20set/Interpolated_CO2_dataset.csv'\
                , index_col='Year',parse_dates=True)
        model_data.index = pd.DatetimeIndex(model_data.index.values,freq=model_data.index.inferred_freq)
        
        size = -49
        train_data = model_data[:size]
        test_data = model_data[size:]

        # ARIMA Model
        y_hat_ar = model_data.copy()
        pred = monthly_model_test.forecast(49)
        y_hat_ar['arima_forecast'] = monthly_model_test.predict(model_data.index.min(), model_data.index.max())


        with tab2:
            st.write(f'Splitting Train and Test Data on Raw data. Leaving Test Data with 5 Years of Time Series. We are going to forecast for the last 5 years, that is from 1994 to 2014.')

            train_data.CO2.plot(legend=True,label='TRAIN',color='green')
            test_data.CO2.plot(legend=True,label='TEST', figsize=(18,6),color='red')
            plt.xlabel('Year', fontsize= 12)
            plt.legend()
            st.pyplot()

            plt.figure(figsize=(16,4))
            test_data.CO2.plot(label="org")
            plt.title("Test data Series", fontsize=14)
            st.pyplot()
            plt.figure(figsize=(16,4))
            test_data["CO2"].rolling(5).mean().plot(label=str(5))
            plt.title("Moving Average "+str(5), fontsize=14)
            plt.legend(loc='best')
            st.pyplot()

        with tab3:

            # Testing vs forecasted 
            plt.figure(figsize=(12,6), dpi=200)
            plt.plot(test_data, label='Test')
            plt.plot(pred, label='Auto regression forecast (ARIMA)')
            plt.legend(loc='best')
            st.pyplot()

        with tab5:
            #Calculating Absolute Percent Error and Error
            # Computing the absolute percent error
            col1, col2 = st.columns(2)
            with col1:
                APE=100*(abs(y_hat_ar['CO2']-y_hat_ar['arima_forecast'])/y_hat_ar['CO2'])
                Error = (abs(y_hat_ar['CO2']-y_hat_ar['arima_forecast']))

                # adding absolute percent error to table
                y_hat_ar['Error']=Error
                y_hat_ar['Absolute Percent Error %']=APE

                y_hat_ar
            with col2:
                #Visualizing the Relationship between the Actual and Predicted ValuesModel Validation
                plt.figure(figsize=(12,8))
                plt.xlabel("Actual Values", fontsize =12)
                plt.ylabel("Predicted values", fontsize =12)
                plt.title("The Scatterplot of Relationship between Actual Values and Predictions", fontsize =16, fontweight = 'bold')
                plt.scatter(y_hat_ar['CO2'], y_hat_ar['arima_forecast'])
                st.pyplot()

        with tab4:
            # Error Evaluation
            monthly_model_test.plot_diagnostics(figsize=(16,8))
            st.pyplot()

        with tab1:

            # Accuracy vs Error Rate

            plt.figure(figsize=(12,8)) 
            # Setting size in Chart based on 
            # given values
            Error_Rate = np.mean(APE[size:])
            Accuracy = 100-np.mean(APE[size:])
            sizes = [Accuracy, Error_Rate]
              
            # Setting labels for items in Chart
            labels = ['Accuracy', 'Error']
              
            # colors
            colors = ['#007500', '#FF0000']
              
            # explosion
            explode = (0.0, 0.05)
              
            # Pie Chart
            plt.pie(sizes, colors=colors, labels=labels,
                    autopct='%1.1f%%', shadow=True,
                    pctdistance=0.85, 
                    explode=explode,
                    startangle=0.0,
                    textprops = {'size':'large',
                               'fontweight':'bold',
                                'rotation':'0.0',
                               'color':'black'})
              
            # draw circle
            centre_circle = plt.Circle((0.0, 0.0), 0.70, fc='white')
            fig = plt.gcf()
              
            # Adding Circle in Pie chart
            fig.gca().add_artist(centre_circle)

            # Adding Title of chart
            plt.title('ARIMA Model \n Accuracy and Error Rate \n on Test Data', fontsize = 16, fontweight = 'bold')
              
            # Add Legends
            plt.legend(labels, loc="upper right")
              
            # Displaying Chart
            st.pyplot()

if nav == "Yearly Forecasting":

    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1)

    st.title("Forecasting $CO_2$ Emission Yearly")
    
    st.write(f"Note: For Different Results Select different Year from the left sidebar")
    st.sidebar.subheader("Select the Year to Forecast from 2015")
    years = st.sidebar.slider("",2015,2026,step = 1)

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
    
    with st.spinner('Wait for it...'):
       time.sleep(2)
    
    pred_yearly = pd.DataFrame()
    pred_yearly['CO2 Emission'] = yearly_model.forecast(year)
      
    st.sidebar.subheader(f"$CO_2$ Emission for the Year {lists}")
    st.sidebar.write(pred_yearly[-1:])
    st.write(f"$CO_2$ Emission Forecasted till the year {lists}" ) 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_yearly.index,y=pred_yearly['CO2 Emission']))

    col1, col2 = st.columns(2)
    with col1:
       col1.write(f'Forecasted Data')
       col1.write(pred_yearly)
    with col2:
       col2.write(f"Interactive Line plot Forecasting for the next {year} Years")
       col2.plotly_chart(fig)


if nav == "Monthly Forecasting":

    my_bar = st.progress(0)

    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1)

    st.title("Forecasting $CO_2$ Emission Monthly")
    
    st.write(f"Note: For Different Results Select different Month from the left sidebar")
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
    months=1
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
    
    pred_monthly = pd.DataFrame()
    pred_monthly['CO2 Emission'] = monthly_model.forecast(months)

    with st.spinner('Wait for it...'):
       time.sleep(2)
   
    st.sidebar.write(f"$CO_2$ Emission for {drop}")
    st.sidebar.write(pred_monthly[-1:])
    st.write(f"$CO_2$ Emission Forecasted till {drop}" )
       
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_monthly.index,y=pred_monthly['CO2 Emission']))

    col1, col2 = st.columns(2)
    with col1:
       col1.write(f'Forecasted Data')
       col1.write(pred_monthly)
    with col2:
       col2.write(f"Interactive Line plot Forecasting for the next {months} Months")
       col2.plotly_chart(fig)
