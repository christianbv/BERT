from datetime import datetime
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import datetime as dt
from dataverk import Client, Datapackage
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.insert(1, "../lib")
from text_processer import *
import plotly.io as plio
import ast
class classifier_datapakke():
    def __init__(self, df: pd.DataFrame, succes_col = "Fikk du gjort det du kom hit for å gjøre", date_col = "Date Submitted"):
        self._initialised    = time.time()
        print("Starting init")
        self._df             = df
        self._succes_col     = succes_col
        self._date_col       = date_col
        self._df["index"]    = df.index.values   
        self._df[succes_col] = self._df[succes_col].apply(lambda x: 1 if x == "Ja" else 0)        
        self._df_matrix      = convert_to_matrix(df)
        self._df_labels      = self.__group_by_label()
        print("Finished init took {0} seconds".format(round(time.time() - self._initialised, 3)))
    def show_texts(self,label, df = None):
        if df is None:
            df = self._df
        index = self.__filter_on_label(label)
        df_s = df.loc[index,:]
        if len(df_s) == 0:
            print("Ingen henvendelser")
        for pred,text in zip(df_s["Prediction"],df_s["raw_text"].values):
            print("\nHENVENDELSE:{0} \nPREDICTIONS:{1}".format(text,pred ))
    def pie_chart_co_occuring_labels(self, df = None, min_Antall = 2):
        if df is None:
            df = self._df
            dist = self._df_labels
        else:
            dist = self.__group_by_label(df)
        labels = dist["Labels"].values
        labels = sorted(labels)
        labels.pop(0)
        sub_data = []
        for label in tqdm(labels):
            p = self.__antall_co_occurences(label, self._df_matrix)
            maks = p.loc[label][0]
            p = p.loc[p["Antall"]> min_Antall]
            p = p.loc[p["Antall"]< maks]
            p["Fullføringsgrad"] = p["Labels"].apply(self.__success_rate)
            p["Fullføringsgrad"] = p["Fullføringsgrad"].apply(lambda x: round(x * 100,4))

            sub_trace = go.Pie(labels=p["Labels"], values=p["Antall"],text = p["Fullføringsgrad"],visible = True if label == "Aktivitetsplan" else False,textinfo='label+percent',
                               insidetextorientation='radial',hovertemplate = "%{label} <br>Antall: %{value} </br>Fullføringsgrad: %{text}%")
            sub_data.append(sub_trace)
            buttons = []
        for i, label in enumerate(labels):
            visibility = [i==j for j in range(len(labels))]
            button = dict(
                             label =  label,
                             method = 'update',
                             args = [{'visible': visibility}])
            buttons.append(button)

        updatemenus = list([
                dict(active = 0,
                     x = 0.17,
                     y=1.08,
                     showactive = True,
                     buttons=buttons)
            ])

        layout = dict(xaxis= dict(title = "x"),
                      yaxis = dict(title = 'y'),
                      height=600,
                      width = 900,
                      showlegend=True,
                      updatemenus=updatemenus,
                      margin_t = 90, #increase the margins to have space for captions
                      annotations = [dict(x=0,
                                                xanchor = "left",
                                                y=1.06,
                                                showarrow=False,
                                                text="Topic:",
                                                xref="paper",
                                                yref="paper")])
        fig = go.Figure(sub_data,layout)
        return fig 

    def pie_chart_monthly(self):
        sub_data = []
        current_month = dt.datetime.now().month + 1
        months = range(2,current_month)
        for month in tqdm(months):
            df_month = self.__filter_on_month(month)
            p = self.__group_by_label(convert_to_matrix(df_month))
            min_antall = int(0.1*p["Antall"].max())
            p = p.loc[p["Antall"]> min_antall].reset_index(drop=True)
           # p["Fullføringsgrad"] = p["Labels"].apply(lambda x: self.__success_rate(x, df_month))
            p["Fullføringsgrad"] = p["Fullføringsgrad"].apply(lambda x: round(x * 100,4))

            sub_trace = go.Pie(labels=p["Labels"], values=p["Antall"],text = p["Fullføringsgrad"],visible = True if month == months[0] else False,textinfo='label+percent',
                               insidetextorientation='radial',hovertemplate = "%{label} <br>Antall: %{value} </br>Fullføringsgrad: %{text}%")
            sub_data.append(sub_trace)
            buttons = []
        labels = months
        for i, label in enumerate(labels):
            visibility = [i==j for j in range(len(labels))]
            button = dict(
                             label =  label,
                             method = 'update',
                             args = [{'visible': visibility}])
            buttons.append(button)

        updatemenus = list([
                dict(active = 0,
                     x = 0.17,
                     y=1.08,
                     showactive = True,
                     buttons=buttons)
            ])

        layout = dict(xaxis= dict(title = "x"),
                      yaxis = dict(title = 'y'),
                      height=600,
                      width = 900,
                      showlegend=True,
                      updatemenus=updatemenus,
                      margin_t = 90, #increase the margins to have space for captions
                      annotations = [dict(x=0,
                                                xanchor = "left",
                                                y=1.06,
                                                showarrow=False,
                                                text="Month:",
                                                xref="paper",
                                                yref="paper")])
        fig = go.Figure(sub_data,layout)
        return fig 
    def get_df_intent(self):
        dist = self.__group_by_label()
        df_hensikt = dist.loc[dist["Labels"].isin(intents)].reset_index(drop = True)
        df_hensikt.columns = ["Hensikt", "Antall", "Fullføringsgrad"]
        df_hensikt = df_hensikt.sort_values(by = "Antall", ascending = False)
        return df_hensikt
    def get_df_theme(self):
        dist = self.__group_by_label()
        df_tema = dist.loc[~dist["Labels"].isin(intents)].reset_index(drop = True)
        df_tema.columns = ["Oppgave/tema", "Antall", "Fullføringsgrad"]
        df_tema = df_tema.sort_values(by = "Antall", ascending = False)
        return df_tema
    
    
    def set_up_data_package(self):
        self._dv = Client()        
        first_published = dt.datetime(year = 2020, month = 8, day = 21)
        now = dt.datetime.now()
        readme = f""" Dette er resultatene fra en klassifiseringsmodell. 
    Modellen er et såkalt LSTM-nettverk (Long short-term memory) som er trent opp på omtrent 2300 annoterte henvendelsetekster.
    Disse henvendelstekstene kommer fra Hotjar spørreundersøkelsen os
    Klassene, eller annotasjone, består av en blanding av 'hensikter' og 'temaer'.
    Hver henvendelse er blitt annotert minst 2 ganger (til en hensikt og et tema), men også noen ganger så består en henvendelse av flere tekster 
         """
        metadata = {
    'title': 'Klassifisering av henvendelsetekster til nav fra Hotjar.no',
    'description': 'Resultatet fra predikeringene til et LSTM nettverk på friteksthenvendelsene fra Hotjar.',
    'readme': readme,
    'accessRights': 'Open', 
    'issued': first_published.isoformat(),
    'modified': now.isoformat(),
    'language': 'Norsk', 
    'periodicity': 'Daglig',
    'temporal': {'from': first_published.strftime('%Y-%m-%d') , 
                 'to': now.strftime('%Y-%m-%d')},
    'author': 'Wilhelm Støren og Christian Vennerød',
    'publisher': {'name': 'Arbeids- og velferdsetaten (NAV)', 
                  'url': 'https://www.nav.no'}, 
    'contactpoint': {'name': 'Wilhelm Støren', 
                     'email': 'wilhelm.storen@nav.no'},
    'license': {'name': 'CC BY 4.0', 
                'url': 'http://creativecommons.org/licenses/by/4.0/deed.no'},
    'keyword':  ["Tekstanalyse", "Nlp", "Hotjar", "Nav.no"],
    'spatial': 'Norge',
    'theme': ['Åpne data'],
    'type': 'datapackage',
    'format': 'datapackage',
    'category': 'category',
    'provenance': 'NAV',
    'store': 'nais',
    'project': 'odata',
    'bucket': 'nav-opendata'
    }
        self._dp = Datapackage(metadata)
        return self._dp
    def update_data_package(self):
        df_hensikt = self.get_df_intent()
        df_tema    = self.get_df_theme()
        pie_chart_monthly = self.pie_chart_monthly()
        pie_chart_labels  = self.pie_chart_co_occuring_labels()
        
        self.set_up_data_package()
        self.df_to_data_package(df_hensikt, "df_hensikt",title = "Fordelingen av hensikten til brukeren", dec ='Tabellen viser hvor mange henvendelser som er klassifisert til de forskjellige hensiktene')
        self.df_to_data_package(df_tema, "df_tema",title = "Fordelingen av oppgaven/temaet til brukeren", dec ='Tabellen viser hvor mange henvendelser som er klassifisert til de forskjellige oppgaven/temaene')
        self.figure_to_data_package(pie_chart_monthly, title = 'Fordeling av oppgaver basert på måned', dec = "Sektordiagrammet viser fordelingen av oppgaver basert på en gitt måned")
        self.figure_to_data_package(pie_chart_labels, title = 'Samtidige forekommende oppgaver til en bestemt oppgave', dec = "Sektordiagrammet viser fordelingen av oppgaver som forekommer sammen med en bestemt oppgave. Dvs at for f. eks 'Tekniske problemer' så vil sektordiagrammet vise fordelingen av oppgaver som forekommer når 'Tekniske problemer' forekommer")
        self.publish_data_package()
        
    
    def figure_to_data_package(self, fig, title, dec):
        self._dp.add_view(spec_type = 'plotly',
            name = '',
            title = title, #'Fordeling av oppgaver basert på måned',
            description = dec,
            spec  = plio.to_json(fig),
            resources = ''
           )
    def df_to_data_package(self, df, name, title, dec):
        self._dp.add_resource(df,
                name,
                spec = {'hidden': False}
               )
        
        self._dp.add_view(spec_type = 'table',
            resources = name,
            name = title,
            title = title,
            description = dec #'Tabellen viser hvor mange henvendelser som er klassifisert til de forskjellige predefinerte oppgavene'
           )
    def publish_data_package(self):
        self._dv.publish(self._dp)
    def __group_by_label(self, df = None):
        if df is None:
            df = self._df_matrix
        
        dist = df.sum()
        dist = pd.DataFrame(dist, columns = ["Antall"])
        dist["Labels"] = dist.index.values
        dist = dist.reset_index(drop = True)
        columns_titles = ["Labels","Antall"]
        dist=dist.reindex(columns=columns_titles)
        dist["Fullføringsgrad"] = dist["Labels"].apply(self.__success_rate)
        return dist
    def __success_rate(self,label, df = None):
        if df is None:
            df = self._df
            df_matrix = self._df_matrix
        else:
            df_matrix = convert_to_matrix(df)
        col = self._succes_col
        index = self.__filter_on_label(label,df_matrix)
        
        dfs = df.loc[index,:]
        tmp = dfs.groupby(by = col).count()
        if len(tmp.index.values) == 1:
            return tmp.index.values[0]
        else:
            ja = tmp.loc[1,"index"]
            nei = tmp.loc[0,"index"]
            fgrad = round( (ja/(ja+nei)),3)
            return fgrad
    def __filter_on_label(self,labels, df_matrix = None):
        if df_matrix is None:
            df_matrix = self._df_matrix
        if type(labels) == list:
            label = labels[0]
        else:
            label = labels
        ind = (df_matrix[df_matrix[label] == 1].index)
        index = np.zeros(len(ind))
        for j,i in enumerate(ind):
            index[j] = int(i)
        if type(labels) == list and len(labels) > 1:
            out = []
            for lab in labels[1:]:
                temp = (df_matrix[df_matrix[lab] == 1].index)
                tempp = np.zeros(len(temp))
                for j,i in enumerate(temp):
                    tempp[j] = i
                out.extend([int(x) for x in index if x in tempp])
            return out
        else:
            return index
    def __antall_co_occurences(self, label, df_labels = None):
        if df_labels is None:
            df_labels = self._df_matrix
        Antall = 0
        index = self.__filter_on_label(label, df_labels)
        df_label = df_labels.loc[index,:]
        temp = pd.DataFrame(df_label.sum(), columns = ["Antall"])
        temp["Labels"] = temp.index
        return temp
    def __filter_on_time(self,start, end, col = "Date Submitted"):
        dfd = pd.DataFrame.copy(self._df)
        if type(dfd[col].values[0]) is str:
            dfd[col] = dfd[col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )
            dfd[col] = dfd[col].dt.strftime("%Y-%m-%d")
        if len(start+end) != 20:
            print("Start and end dates should be on the format 'yyyy-mm-dd'")
        else:
            out = dfd[(dfd[col] > start) & (dfd[col] <= end)]
            return out
    
    def __filter_on_month(self,month):
        next_month = str(month + 1)
        month = "0"+str(month)
        next_month = "0"+next_month
        month = "2020-"+month
        next_month = "2020-"+next_month
        start = month + "-01"
        end   = next_month + "-01"
        return self.__filter_on_time(start, end)

def convert_to_matrix(df, keep = []):
    df = pd.DataFrame.copy(df)
    labels = set()
    for row in df[["Prediction","index"]].values:
        label = row[0]
        for l in label:
            labels.add(l)

    # Creating matrix of labels
    labels = list(labels)
    for label in labels:
        df[label] = 0

    col_inds = df.columns.values
    for index,val in enumerate(df[["Prediction"]].values):
        labels = val[0]
        for label in labels:
            col_ind = np.where(col_inds == label)[0]
            df.iloc[index,col_ind] = 1
    garbage = ["Date Submitted","Hva kom du hit for å gjøre","Fikk du gjort det du kom hit for å gjøre","Hvor lang tid brukte du?","raw_text","Prediction", "index"]
    garbage = [gar for gar in garbage if gar not in keep]
    df.drop(garbage, axis = 1, inplace = True)
    return df
