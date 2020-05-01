import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

from datetime import datetime

import plotly.offline as py
import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def data_estado(estado, df):
    dfe = df.loc[df["Estado"] == estado]
    dfe.loc[dfe["Casos Acumulados"] == 0, "Casos Acumulados"] = None
    dfe.dropna(inplace=True)
    dfe.reset_index(drop=True, inplace=True)
    return dfe


def data_regiao(regiao, df):
    dfr = df.loc[df["Região"] == regiao]
    dfr.loc[dfr["Casos Acumulados"] == 0, "Casos Acumulados"] = None
    dfr.dropna(inplace=True)
    dfr.reset_index(drop=True, inplace=True)
    return dfr


def plot_estado(dfe, estado, color):
    trace = [go.Bar(x = dfe["Data"], 
                y = dfe["Casos Acumulados"],
                marker = {"color":"{}".format(color)},
                opacity=0.8)]

    layout = go.Layout(title='Casos de COVID-19 - {}'.format(estado),
                    yaxis={'title':"Número de casos"},
                    xaxis={'title': 'Data do registro'})

    fig = go.Figure(data=trace, layout=layout)
    return(fig)


def plot_regiao(dfr, regiao):

    estados = list(dfr["Estado"].unique())
    trace = []
    i = 0
    for estado in estados:
        color = 123456+i
        dfe = data_estado(estado, dfr)
        trace.append(go.Bar(x = dfe["Data"], 
                    y = dfe["Casos Acumulados"],
                    name="{}".format(estado),
                    marker = {"color":"#{}".format(color)},
                    opacity=0.6))
        i += 98765
        
    layout = go.Layout(title='Casos de COVID-19 - {}'.format(regiao),
                    yaxis={'title':"Número de casos"},
                    xaxis={'title': 'Data do registro'},
                    barmode="stack")

    fig = go.Figure(data=trace, layout=layout)
    return(fig)


def TimeStempToStr(ts):
    try:
        aux1 = datetime.strptime(str(ts), '%Y-%m-%d %H:%M:%S') 
        aux2 = aux1.strftime('%m/%d/%Y').split("/")
        aux3 =  aux2[2]+"-"+aux2[0]+"-"+aux2[1]
        return aux3
    except:
        return ts

@st.cache()
def plot_previsao(df, local, color1, color2):
    ncasos = df.groupby("Data").sum()["Casos Acumulados"].values
    date = df.groupby("Data").sum()["Casos Acumulados"].index

    dfcasos = pd.DataFrame(data=ncasos,index=date,columns=["Número de Casos"])
    dfcasos.loc[dfcasos["Número de Casos"] == 0, "Número de Casos"] = None
    dfcasos.dropna(inplace=True)

    full_scaler = MinMaxScaler()
    scaled_full_data = full_scaler.fit_transform(dfcasos)

    length = 12
    n_features = 1
    generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)

    model = Sequential()
    model.add(LSTM(100,activation="relu",input_shape=(length,n_features))) # can add dropout too
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(generator,epochs=75)


    forecast = []
    forecast_period = 10
    first_eval_batch  = scaled_full_data[-length:]
    current_batch = first_eval_batch.reshape((1,length,n_features))

    for i in range(forecast_period):
        
        # get prediction 1 time atamp ahead ([0] is for grabbing just the number insede the brackets)
        current_pred = model.predict(current_batch)[0]
        
        # store prediction
        forecast.append(current_pred)
        
        # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        

    forecast = full_scaler.inverse_transform(forecast)
    forecast_index = pd.date_range(start="2020-04-30", periods=forecast_period, freq="D")
    forecast_df = pd.DataFrame(data=forecast,index=forecast_index,columns=["Forecast"])

    dfresult = pd.concat([dfcasos, forecast_df])
    dfresult.reset_index(inplace=True)
    dfresult["Data"] = dfresult["index"].apply(TimeStempToStr)



    trace = [go.Bar(x = dfresult["Data"], 
                y = dfresult["Número de Casos"],
                name = 'Casos reais',
                marker = {"color":color1},
                opacity=0.8), 
                
            go.Bar(x = dfresult["Data"], 
                y = dfresult["Forecast"],
                name = 'Previsão',
                marker = {"color":color2},
                opacity=0.8)]

    layout = go.Layout(title='Previsão de COVID-19 no {}'.format(local),
                    yaxis={'title':"Número de casos"},
                    xaxis={'title': 'Data do registro'})

    fig2 = go.Figure(data=trace, layout=layout)

    return fig2


@st.cache()
def plot_previsao_estado(df, estado, color1, color2):
    dfe = data_estado(estado, df)

    return plot_previsao(dfe, estado, color1, color2)

@st.cache()
def plot_previsao_regiao(df, regiao, color1, color2):
    dfr = data_regiao(regiao, df)

    return plot_previsao(dfe, regiao, color1, color2)



    # ncasos = dfe.groupby("Data").sum()["Casos Acumulados"].values
    # date = dfe.groupby("Data").sum()["Casos Acumulados"].index

    # dfcasos = pd.DataFrame(data=ncasos,index=date,columns=["Número de Casos"])
    # dfcasos.loc[dfcasos["Número de Casos"] == 0, "Número de Casos"] = None
    # dfcasos.dropna(inplace=True)

    # full_scaler = MinMaxScaler()
    # scaled_full_data = full_scaler.fit_transform(dfcasos)

    # length = 12
    # n_features = 1
    # generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)

    # model = Sequential()
    # model.add(LSTM(100,activation="relu",input_shape=(length,n_features))) # can add dropout too
    # model.add(Dense(1))
    # model.compile(optimizer="adam", loss="mse")
    # model.fit(generator,epochs=75)


    # forecast = []
    # forecast_period = 10
    # first_eval_batch  = scaled_full_data[-length:]
    # current_batch = first_eval_batch.reshape((1,length,n_features))

    # for i in range(forecast_period):
        
    #     # get prediction 1 time atamp ahead ([0] is for grabbing just the number insede the brackets)
    #     current_pred = model.predict(current_batch)[0]
        
    #     # store prediction
    #     forecast.append(current_pred)
        
    #     # update batch to now include prediction and drop first value
    #     current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        

    # forecast = full_scaler.inverse_transform(forecast)
    # forecast_index = pd.date_range(start="2020-04-30", periods=forecast_period, freq="D")
    # forecast_df = pd.DataFrame(data=forecast,index=forecast_index,columns=["Forecast"])

    # dfresult = pd.concat([dfcasos, forecast_df])
    # dfresult.reset_index(inplace=True)
    # dfresult["Data"] = dfresult["index"].apply(TimeStempToStr)



    # trace = [go.Bar(x = dfresult["Data"], 
    #             y = dfresult["Número de Casos"],
    #             name = 'Casos reais',
    #             marker = {"color":"#ffd700"},
    #             opacity=0.8), 
                
    #         go.Bar(x = dfresult["Data"], 
    #             y = dfresult["Forecast"],
    #             name = 'Previsão',
    #             marker = {"color":"#9acd32"},
    #             opacity=0.8)]

    # layout = go.Layout(title='Previsão de COVID-19 por estado - {}'.format(estado),
    #                 yaxis={'title':"Número de casos"},
    #                 xaxis={'title': 'Data do registro'})

    # fig2 = go.Figure(data=trace, layout=layout)

    # return fig2





####################################################################
st.title("COVID-19 Brasil")


df = pd.read_csv("arquivo_geral.csv", sep=";")
df.rename(columns={"regiao":"Região", "estado":"Estado", "data":"Data", 
            "casosNovos": "Casos Novos", "casosAcumulados":"Casos Acumulados", 
            "obitosNovos":"Obitos Novos", "obitosAcumulados":"Obitos Acumulados"}, inplace=True)

siglas_estados = list(df["Estado"].unique())
#siglas_estados.insert(0,None)

siglas_regiao = list(df["Região"].unique())

###############################################################
st.title("Casos confirmados por região")

regiao = st.selectbox("Selecione uma regiao: ", siglas_regiao)
dfr = data_regiao(regiao, df)

if st.checkbox("Mostrar todos os dados de {}".format(regiao)):
    st.table(dfr)

figr = plot_regiao(dfr, regiao)
st.plotly_chart(figr)



###############################################################
st.title("Casos confirmados por estados")

estado = st.selectbox("Selecione um estado: ", siglas_estados)
dfe = data_estado(estado, df)

if st.checkbox("Mostrar todos os dados de {}".format(estado)):
    st.table(dfe)

fige = plot_estado(dfe, estado, "#ffa07a")
st.plotly_chart(fige)










######################################################
st.title("Previsão de casos no Brasil")
st.warning("A previsão pode demorar alguns segundos.")
if st.checkbox("Plotar previsão"):
    
    st.plotly_chart(plot_previsao(df, "Brasil", "#a020f0", "#ff1493"))

#####################################################
st.title("Previsão de casos por região")
st.warning("A previsão pode demorar alguns segundos.")
prev_regiao = st.selectbox("Selecione uma regiao", siglas_regiao)
if st.checkbox("Plotar previsão por regiao"):
    
    st.plotly_chart(plot_previsao_regiao(df, prev_regiao, "#008b8b", "#cd5c5c"))



#####################################################
st.title("Previsão de casos por estado")
st.warning("A previsão pode demorar alguns segundos.")
prev_estado = st.selectbox("Selecione um estado", siglas_estados)
if st.checkbox("Plotar previsão por estado"):
    
    st.plotly_chart(plot_previsao_estado(df, prev_estado, "#ffd700", "#9acd32"))






