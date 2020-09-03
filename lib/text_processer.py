import pandas as pd
import numpy as np
import re
import json
import spacy
import os
import os.path
nlp = spacy.load("../hente_brukerhenvendelser/spacy_norsk_custom")

import requests
import math
import nltk
from nltk.corpus import stopwords
import timeit
import warnings

from filtering_methods import *
from grammar import *

from io import StringIO
from dataverk import Client, Datapackage
from dataverk_vault.api import set_secrets_as_envs as set_secrets_as_envs
set_secrets_as_envs()    
username = os.environ['HOTJAR_USERNAME']
password = os.environ['HOTJAR_PASSWORD']
from datetime import date
today = date.today().strftime("%Y-%m-%d")

import multiprocessing as mp
from multiprocessing import Pool

####################################################################################################################

KORONAVEIVISER = ("KORONAVEIVISER","https://insights.hotjar.com/api/v1/sites/118350/polls/490813/responses?format=csv","koronaveiviser.csv","Hva kom du hit for å gjøre")
TOPPOPPGAVER   = ("TOPPOPPGAVER","https://insights.hotjar.com/api/v1/sites/118350/polls/484397/responses?format=csv","toppoppgaver.csv","Hva kom du hit for å gjøre")
DITTNAV        = ("DITTNAV","https://insights.hotjar.com/api/v1/sites/118350/polls/505021/responses?format=csv","dittnav.csv","Hva kom du hit for å gjøre")
SKJEMAVEILEDER = ("SKJEMAVEILEDER","https://insights.hotjar.com/api/v1/sites/118350/polls/385837/responses?format=csv","skjemaveileder.csv","FOO SETT INN")

# Instansierer globalt for å unngå at hver delprosess starter egen instans
#misspellings = misspellings()

stoppwords = set(pd.read_pickle("../data/stoppord_nav"))


import spacy
nlp = spacy.load("nb_core_news_sm")
nltk.download('stopwords')


# UPDATE FUNCTIONS
##################################################################################################################


def get_all_surveys():
    return KORONAVEIVISER,TOPPOPPGAVER,DITTNAV,SKJEMAVEILEDER

def update_all_data():
    '''
    1. Downloads all new raw data
    2. Updates all cleaned data
    3. TODO - preprocesses all new data
    '''
    for survey in [KORONAVEIVISER,TOPPOPPGAVER,DITTNAV,SKJEMAVEILEDER]:
        processer = text_processer(survey)
        processer.update_raw_data()
        processer.update_cleaned_data()
        #processer.update_processed_data()


# TEXT-PROCESSOR CLASS:
####################################################################################################################

class text_processer():
    
    def __init__(self,survey = TOPPOPPGAVER):
        self.survey = survey[0]
        self.url = survey[1]
        self.end_path = survey[2]
        self.column_text = survey[3]
        self.raw_data = None
        self.cleaned_data = None
        self.processed_data = None

        # Standard
        self.base_path = '../data/csv/'

        
    def get_raw_data(self) -> pd.DataFrame:
        '''
        1. Loads in the current chosen raw survey as a CSV file, converts it to pd.DataFrame, changes type of date column to datetime
        2. sets internal variable self.raw_data and returns it
        '''
        df = pd.read_csv(self.__get_paths(False))
        df = convert_column_to_date(df,"Date Submitted")
        self.raw_data = df
        return df
    
    def get_cleaned_data(self) -> pd.DataFrame:
        '''
        1. Loads in the current chosen cleaned survey as a CSV file, converts it to pd.DataFrame, changes type of date column to datetime
        2. sets internal variable self.cleaned_data and returns it
        '''
        df =  pd.read_csv(self.__get_paths(True))
        df = convert_column_to_date(df,"Date Submitted")
        self.cleaned_data = df
        return df
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        '''
        Loads in the current chosen preprocessed survey as a CSV file, converts it to dataframe, sets internal variable self.processed_data and returns it
        '''
        path = self.base_path+"preprocessed_polls/"+self.end_path
        
        if not os.path.isfile(path):
            print("Processed file does not exist. Returns cleaned data and sets self.processed_data = self.cleaned_data!")
            df = self.get_cleaned_data()
            self.processed_data = df
            self.has_processed = False
            return df
        else:
            df = pd.read_csv(self.base_path+"preprocessed_polls/"+self.end_path)
            self.processed_data = df
            self.has_processed = True
            return df
    
    def update_raw_data(self):
        '''
        1. Downloads the entire survey from hotjar
        2. Replaces it with the current rawdata file
        TODO: set types for the different columns.
        '''
        df = self.__get_hotjar_data()
        name_csv = str(self.end_path).lower()
        print(name_csv)
        df.to_csv(self.base_path+'raw_polls/'+name_csv, sep = ',', encoding = 'utf-8', index = False)
        print("Updated the raw survey:",self.survey)

    def update_cleaned_data(self):
        '''
        Uses functions in tekstprosessering.py..
        1. Loads in the survey from raw_data file
        2. Removes columns as specified
        3. Replaces it with the current cleaned file
        '''
        
        if self.survey == "KORONAVEIVISER":
            clean_KORONAVEIVISER(self.base_path+"raw_polls/"+self.end_path)
            print("Updated the cleaned survey: ",self.survey)
        elif self.survey == "TOPPOPPGAVER":
            clean_TOPPOPPGAVER(self.base_path+"raw_polls/"+self.end_path)
            print("Updated the cleaned survey:",self.survey)
        
        elif self.survey == "DITTNAV":
            clean_DITTNAV(self.base_path+"raw_polls/"+self.end_path)
        else:
            raise ValueError("Not implemented yet")    
    
    def update_preprocessed_data(self, tokenizer = True, use_stopwords = True, check_grammar = False, use_lemmatization = False):
        '''
        1. Loads in the updated survey from cleaned file
        2. Loads in the survey from preprocessed file
        3. Finds new instances in cleaned file that are not in preprocessed file
        4. Preprocesses the new instances, on prespecified column name
        5. Updates the preprocessed file
        '''
        
        cleaned = self.get_cleaned_data()
        processed = self.get_preprocessed_data()
        processed = convert_column_to_date(processed,"Date Submitted")
        
        if not self.has_processed:
            new_instances = cleaned
            self.new_instances = new_instances.iloc[:,:]
        else:
            cleaned["already_processed"] = cleaned["Date Submitted"].isin(processed["Date Submitted"])
            new_instances = cleaned[cleaned["already_processed"] == False].drop("already_processed",axis = 1)
            self.new_instances = new_instances.iloc[:,:]
    
    
        n_cores = mp.cpu_count()
        df_split = np.array_split(self.new_instances, n_cores, axis=0)
        print(len(df_split))
        kwargs = {"col":self.column_text, "stopwords":stoppwords, "grammar": check_grammar}
        self.new_instances = applyParallel(df_split, process_column, kwargs, n_cores)
        
        self.new_instances.dropna(inplace = True)
        
        self.new_instances = convert_column_to_date(self.new_instances, "Date Submitted")
        
        # Appending the raw text next to preprocessed text
        cleaned_texts = new_instances.loc[self.new_instances.index.values,:]
        self.new_instances["raw_text"] = cleaned_texts["Hva kom du hit for å gjøre"].values
        print(self.new_instances.shape)
        
        if not self.has_processed:
            processed = self.new_instances
        else:
            # Appending the new instances to the current, earlier processed instances
            processed = processed.append(self.new_instances)
            processed = convert_column_to_date(processed,"Date Submitted")

        # Saving to csv
        #self.new_instances.to_csv(self.base_path + "preprocessed_polls/"+self.end_path, index = False)
        processed.to_csv(self.base_path+"preprocessed_polls/"+self.end_path, index = False)
        print("Updated the processed survey:",self.survey)
        
        #misspellings.save_misspellings()
        print("Updated misspellings_list")
        
        return
        
    def __get_hotjar_data(self):
        '''
        Loads all raw data for current survey from hotjar
        '''
    
        url = self.url 
        set_secrets_as_envs()    
        username = os.environ['HOTJAR_USERNAME']
        password = os.environ['HOTJAR_PASSWORD']
        loginurl = "https://insights.hotjar.com/api/v2/users"

        auth = json.dumps({"action": "login", "email": username, "password": password,  "remember": True})
        content_type = "application/json"

        
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/75.0.3770.90 Chrome/75.0.3770.90 Safari/537.36"
        headers = {
        "Content-Type": content_type,
        "user-agent" : user_agent}
        
        
        session = requests.Session()
        session.headers = headers
        
        pollid = url.split("/")[8]
        response = session.post(loginurl,data=auth)
        response.raise_for_status()
                
        data = session.get(url)
        df = pd.read_csv(StringIO(data.text))
        
        if len(df) < 10:
            print(auth)
            print(response)
            print(data)
            print(username)
            print(password)
            raise TypeError("FEIL VED TILKOBLING TIL HOTJAR!..")
        return df            
        
    def get_end_path(self):
        return self.end_path
                           
    def __get_paths(self,cleaned = False):
        '''
        Helper method to get the path for the current survey chosen
        '''
        path = self.base_path
        if cleaned:
            path += 'cleaned_polls/'
        else:
            path += 'raw_polls/'
        return path + self.end_path
    
        
        
        
        

        
# MULTIPROCESSING HELPER FUNCS
#######################################################################################################################

# For multiprocessing - nestede klasser fungerer ikke.
# Map does not allow multiple parameters so we create a wrapper class
class WithExtraArgs(object):
    def __init__(self, func, **args):
        self.func = func
        self.args = args
    def __call__(self, df):
        return self.func(df, **self.args)

def applyParallel(df_split, func, kwargs, cores):
    pd.options.mode.chained_assignment = 'warn'
    
    with Pool(cores) as p:
        ret = p.map(WithExtraArgs(func, **kwargs),  df_split)
    
    pd.options.mode.chained_assignment = None
    res = pd.concat(ret)
    return pd.concat(ret)


def process_column(df_slice, col:str, stopwords = [], grammar = False):    
    indexes = df_slice.index.values
    
    for ind, doc in enumerate(nlp.pipe(df_slice[col], disable = ["tagger","parser","ner"])):        
        new_doc = []
        for tok in doc:
            if (tok.is_punct or tok.is_space or tok.text.lower() in stopwords):
            #if (tok.is_punct or tok.is_space):
                continue
            if not grammar:
                new_doc.append(lemmatisering(tok.text.lower()))
            else:
                word = misspellings.check_word(tok.text.lower())
                if word != None:
                    new_doc.append(lemmatisering(word.lower()))
        if len(new_doc) < 1:
            new_doc = None
        df_slice.at[indexes[ind],col] = new_doc  
        
    return df_slice

import re
def lemmatize(x):  
    x = x.lower()
    lem = {"permittering": r'(\bperm+it+ering[en]{0,2})',
           "spørsmål": r'(\bspm.?\Z)',
           "permittert": r'(\bperm+it+ert)',
              "utbetaling": r'\butbet+al[ingenrt]*|(\butbet[.]?)',
              "feriepenger": r'\bferiepeng[ern]{0,3}',
              "dagpenger": r'\bdagpeng[aenr]{0,3}',
              "sende": r'\bsend[erdte]{0,3}\Z',
              "meldekort": r'\bmeldekort[etrn]{0,3}',
              "penger": r'\bpeng[a-z]{0,3}',
               "få": r'(\bfå[rnt]{1,2})|(\bfikk)',
               "finne": r'(\bfinne[r]?)',
               "ferie": r'\bferie[ern]*',
               "arbeidsgiver": r'\barbeidsgiver[en]{1,2}\Z',
               "lure": r'(\blurte*)|(\blurer)',
               "jobbe": r'\bjobb[ert]{1,2}\Z', #jobb?
               "arbeide": r'\barbeide[ret]{1,3}',
               "korona": r'(\bcorona[viruset]{0,6}\Z)|(\bkorona[viruset]{0,6}\Z)|(\bcovid[-19]{0,3}\Z)',
                "søknad": r'(\bstøk?nad[en]{0,3}\Z)',
                "rettigheter": r'(\brett?ighet[ern]{0,3})',
               "arbeidssøker": r'(\barbeidss?øker)',
                "registrere": r'(\breg\Z)|(\breg?istrere\Z)',
                "sykemeldt": r'(\bsyke?meld[etr]{1,3})|(sjuke?me?ldt?)',
               "sykemelding": r'(\bsyke?melding[en]{0,2})|(\bsjuke?melding[en]{0,2})',
                "sykepenger": r'(\bsyke?penge[ner]{0,3})|(\bsjuke?penge[ner]{0,3})',
               "lege": r'(\blegen?)',
               "fastlege": r'(\bfastlegen)',
               "melding": r'(\bmld)|(\bmelding[enr]{0,3})\Z',
               "lønn": r'(\blønn[nea]{0,3})',
               "pensjon": r'(\bpensjon[en]{0,2})|(\bpension[en]{0,2})',
               "informasjon": r'(\binformasjon[en]{0,2})|(\binfo)',
               "konto": r'\bkto',
               "kontonummer": r'(\bkontonummer[et]{0,2})|(\bkontonm?r.?)',
               "oppdatere": r'(\boppdaterte?\Z)|(\boppdatering\Z)',
               "dokument": r'(\bdokument[ernt]{0,3}\Z)|(\bdok)(|\bdoc)',
               "aktivitetsplan": r'(\bak?tivitets?plan[en]{0,3})',
               "endre": r'(\bendre)|(\bendring)',
               "ettersende": r'(\bettersende?)|({\bettersending})',
               "vedrørende":r'(\bvedr[.]?)',
                "angående":r'(\bang.)',
                "telefon":r'(\btlf[.]?)',
                "offentlig":r'(\boff.)',
                "konto":r'(\bkto.)',
                "kontonummer":r'(\bkontonr.)',
                "på grunn av ": r'(\bpga.)',
                "nummer":r'(\bnr[.]?)',
                "registrere":r'(\breg.)',
                "tidligere":r'(\btidl[.]?)',
                "med vennlig hilsen":r'(\bmvh[.]?)',
                "arbeids":r'(\barb.)',
                "for eksempel":r'(\bf.eks.)|(\bf. eks.)',
          } 
    for key in lemmas.keys():
        x = re.sub(lemmas[key], key, x)
    return x
    
import re
    
def lemmatisering(tekst):
    lem = {"permittering": r'(\bperm+it+ering[en]{0,2})',
           "spørsmål": r'(\bspm.?\Z)',
           "permittert": r'(\bperm+it+ert)',
              "utbetaling": r'\butbet+al[ingenrt]*|(\butbet[.]?)',
              "feriepenger": r'\bferiepeng[ern]{0,3}',
              "dagpenger": r'\bdagpeng[aenr]{0,3}',
              "sende": r'\bsend[erdte]{0,3}\Z',
              "meldekort": r'\bmeldekort[etrn]{0,3}',
              "penger": r'\bpeng[a-z]{0,3}',
               "få": r'(\bfå[rnt]{1,2})|(\bfikk)',
               "finne": r'(\bfinne[r]?)',
               "ferie": r'\bferie[ern]*',
               "arbeidsgiver": r'\barbeidsgiver[en]{1,2}\Z',
               "lure": r'(\blurte*)|(\blurer)',
               "jobbe": r'\bjobb[ert]{1,2}\Z', #jobb?
               "arbeide": r'\barbeide[ret]{1,3}',
               "korona": r'(\bcorona[viruset]{0,6}\Z)|(\bkorona[viruset]{0,6}\Z)|(\bcovid[-19]{0,3}\Z)',
                "søknad": r'(\bstøk?nad[en]{0,3}\Z)',
                "rettigheter": r'(\brett?ighet[ern]{0,3})',
               "arbeidssøker": r'(\barbeidss?øker)',
                "sykemeldt": r'(\bsyke?meld[etr]{1,3})|(sjuke?me?ldt?)',
               "sykemelding": r'(\bsyke?melding[en]{0,2})|(\bsjuke?melding[en]{0,2})',
                "sykepenger": r'(\bsyke?penge[ner]{0,3})|(\bsjuke?penge[ner]{0,3})',
               "lege": r'(\blegen?)',
               "fastlege": r'(\bfastlegen)',
               "melding": r'(\bmld)|(\bmelding[enr]{0,3})\Z',
               "lønn": r'(\blønn[nea]{0,3})',
               "pensjon": r'(\bpensjon[en]{0,2})|(\bpension[en]{0,2})',
               "informasjon": r'(\binformasjon[en]{0,2})|(\binfo)',
               "konto": r'\bkto',
               "kontonummer": r'(\bkontonummer[et]{0,2})|(\bkontonm?r.?)',
               "oppdatere": r'(\boppdaterte?\Z)|(\boppdatering\Z)',
               "dokument": r'(\bdok)(|\bdoc)',
               "aktivitetsplan": r'(\bak?tivitets?plan[en]{0,3})',
               "endre": r'(\bendre)|(\bendring)',
               "ettersende": r'(\bettersende?)|({\bettersending})',
               "vedrørende":r'(\bvedr[.]?)',
                "angående":r'(\bang.)',
                "telefon":r'(\btlf[.]?)',
                "offentlig":r'(\boff.)',
                "konto":r'(\bkto.)',
                "kontonummer":r'(\bkontonr.)',
                "på grunn av ": r'(\bpga.)',
                "nummer":r'(\bnr[.]?)',
                "registrere":r'(\breg.\Z)',
                "tidligere":r'(\btidl[.]?)',
                "med vennlig hilsen":r'(\bmvh[.]?)',
                "arbeids":r'(\barb.\Z)',
                "for eksempel":r'(\bf.eks.)|(\bf. eks.)',
          } 
    r = False
    if type(tekst) is list:
        tekst = " ".join(item for item in tekst)
        r = True
    for key in lem.keys():
        tekst = re.sub(lem[key], key, tekst)
    if r == True:
        tekst = tekst.split(" ")
    return tekst
