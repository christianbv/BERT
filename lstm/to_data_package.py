import sys
sys.path.insert(1, "../lib")
from datapakke_custom import classifier_datapakke
import pandas as pd
data_path = "../data/"
df = pd.read_pickle(data_path + "pickle/predicted_toppoppgaver/pred.pkl")
dp = classifier_datapakke(df)
dp.update_data_package()
