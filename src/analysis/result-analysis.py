
import pandas as pd
import os
from sql_analytics.src.utils.sql_database import SQLDatabase
from sql_analytics.src.utils.exec_eval import *
from tqdm import tqdm
import json
import numpy as np
import csv

def test_sqls(host, port, database, username, password, data):
    database_uri = f"postgresql://{username}:{password}@{host}:{str(port)}/{database}"
    sqldb = SQLDatabase.from_uri(database_uri)
    sqldb._schema = "experiment"
    tbl = []
    for d in tqdm(data):  # data is a record
        _db_id = d["db_id"]
        _question = d["question"]
        _pred = d["generated_query"]
        _gt = d["query"]
        _hardness = d["hardness"]
        _process_time = d["process_time"]
        _lang = d["lang"]
     #    _Max_Score = d["Max_Score"]
     #    _Avg_Score = d["Avg_Score"]
        hard_label, gt_res, pred_res = sql_eq(
            sqldb,
            _gt,
            _pred,
            order_matters=False,
            with_details=False,
            fetch="many",
            max_rows=50,
        )
        soft_label, gt_res, pred_res = sql_eq(
            sqldb,
            _gt,
            _pred,
            order_matters=False,
            with_details=False,
            fetch="many",
            max_rows=50,
            is_hard=False
        )
        partial_label, gt_res, pred_res = sql_eq(
            sqldb,
            _gt,
            _pred,
            order_matters=False,
            with_details=False,
            fetch="many",
            max_rows=50,
            is_hard=False,
            is_partial=True
        )
        tbl.append(
            {
                "db_id": _db_id,
                "question": _question,
                "query_gt": _gt,
                "query_pred": _pred,
                "res_gt": gt_res[:10],
                "res_pred": pred_res[:10],
                "hard_label": hard_label,
                "soft_label": soft_label,
                "partial_label": partial_label,
                "hardness": _hardness,
                "lang": _lang,
                "process_time": _process_time,
               #  "max_score": _Max_Score,
               #  "avg_score": _Avg_Score,
            }
        )
    return pd.DataFrame(tbl)
def calculate_ex_accuracy(data_file):
     data = data_file.to_dict("records")
     host = '160.85.252.201'
     port = 18001
     database = 'postgres'
     username = 'dbadmin@sdbpstatbot01'
     password = '579fc314a8f73e881a9146901971d5b9'
     database_uri = f"postgresql://{username}:{password}@{host}:{str(port)}/{database}"
     sqldb = SQLDatabase.from_uri(database_uri)
     sqldb._schema = "experiment"

     res = test_sqls(host, port, database, username, password, data)
     
     print(
     f"hard: {round(len(res[res['hard_label'] == True]) / len(res)*100, 2)}%\nsoft: {round(len(res[res['soft_label'] == True]) / len(res)*100, 2)}%\npartial: {round(len(res[res['partial_label'] == True]) / len(res)*100, 2)}%"
     )
     
     result = {"hard":round(len(res[res['hard_label'] == True]) / len(res)*100, 2),
          "soft":round(len(res[res['soft_label'] == True]) / len(res)*100, 2),
          "npartial":round(len(res[res['partial_label'] == True]) / len(res)*100, 2) }

     return result,res

# read the file with correct hardness
# data/statbot/nl2sql_data/final_dev_statbot_old.csv
with open("data/statbot/nl2sql_data/final_dev_statbot_old.csv") as f:
     hardness_df = pd.read_csv(f, delimiter= ',')

hardness_df['question'] = hardness_df['question'].str.replace("\n","").str.strip()
print(len(hardness_df.index))



# look at  the csvs in results for each experiments:
results_path='data/statbot/gpt-3.5-turbo-16k/runs/'  #data/statbot/mixtral/   data/statbot/gpt-3.5-turbo-16k/runs/
#data/statbot/mixtral/mulitlingual-embeddings 
target="zero-shots"
csv_files = [f for f in os.listdir(os.path.join(results_path,target,"csv-outputs")) if  f.endswith('.csv')]
n_times_run_results={i:[] for i in [0]} #1,2,3,4,5,6,8
for csv_file in csv_files:
     #shot = csv_file.split("-")[0]
     shot=0
     csv_file_ = os.path.join(results_path,target,"csv-outputs",csv_file)
     print(csv_file)
     with open(csv_file_) as f:
          d = pd.read_csv(f, delimiter= ',')
     for idx, row in hardness_df.iterrows():
          question = row['question']
          prediction = d.loc[d['question'] == question, 'generated_query'].values
          process_time = d.loc[d['question'] == question, 'process_time'].values
          if len(prediction) > 0:
               hardness_df.at[idx, 'generated_query'] = prediction[0]
               hardness_df.at[idx, 'process_time'] = process_time[0]

     result,res = calculate_ex_accuracy(hardness_df)
     # saving the results

     with open(os.path.join(results_path,target,"csv-outputs-with-labels",csv_file), 'w', newline='') as file:
        
        writer = csv.writer(file)
        writer.writerow(['db_id',"question","query","generated_query","res_gt","res_pred","hard_label","soft_label","partial_label","hardness","lang","process_time"])
     
        for idx, item in res.iterrows():
            writer.writerow([item['db_id'],
                             item["question"],
                             item["query_gt"].strip(),
                              item['query_pred'].strip(),
                              item['res_gt'],
                              item['res_pred'],
                              item['hard_label'],
                              item['soft_label'],
                              item['partial_label'],
                              item['hardness'],
                              item['lang'],
                              item['process_time']
            ]
                              )


     n_times_run_results[int(shot)].append(result)
     print(n_times_run_results)


with open(f"{results_path}/{target}/results.json", "w") as output:
          json.dump(n_times_run_results, output, indent=2, default=int)

with open(f"{results_path}/{target}/results.json") as input:
          d = json.load(input)
          results={k:{}for k in [0]} #
          for shot, rs in d.items():
               hards=[]
               softs=[]
               partials=[]
               for r in rs:
                    hard=r['hard']
                    soft=r['soft']
                    partial=r['npartial']
                    hards.append(float(hard))
                    softs.append(float(soft))
                    partials.append(float(partial))
          
               hard_mean_var={
                    "mean":"%.2f" % np.mean(hards),
                    "std":"%.2f" % np.std(hards)
                    
               }
               
               soft_mean_var={
                    "mean":"%.2f" % np.mean(softs),
                    "std":"%.2f" % np.std(softs)
               }

               partial_mean_var={
                    "mean":"%.2f" % np.mean(partials),
                    "std":"%.2f" % np.std(partials)
               }
               results[int(shot)]={'hards':hard_mean_var,
                              'soft':soft_mean_var,
                              'partial':partial_mean_var}

with open(f"{results_path}/{target}/mean_std.json","w") as llm:
               json.dump(results, llm, indent=2,default=int)
               


# 5-shot-examples-multi-train-70-30-test_old_run#0.csv
# hard: 42.66%
# soft: 48.95%
# partial: 50.35%
     
    #  {
    #   "en": {
    #     "hard": 36.07,
    #     "soft": 39.34,
    #     "partial": 39.34,
    #     "number_of_examples": 61
    #   },
    #   "de": {
    #     "hard": 47.56,
    #     "soft": 54.88,
    #     "partial": 57.32,
    #     "number_of_examples": 82
    #   }
    # },
     

#6-shot_Mixtral-8x7B-Instruct-v0.1-train-test-old-run#0.csv
    # {
    #   "hard": 28.67,
    #   "soft": 35.66,
    #   "npartial": 38.46
    # },