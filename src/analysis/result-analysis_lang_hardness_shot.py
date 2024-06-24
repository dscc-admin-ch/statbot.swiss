
import pandas as pd
import os
from sql_analytics.src.utils.sql_database import SQLDatabase
from sql_analytics.src.utils.exec_eval import *
from tqdm import tqdm
import json
import numpy as np

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
     #    _process_time = d["process_time"]
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
               #  "process_time": _process_time,
               #  "max_score": _Max_Score,
               #  "avg_score": _Avg_Score,
            }
        )
    return pd.DataFrame(tbl)
def calculate_hardness_label(label,data):
     db_lang_dict = data[["db_id", "lang"]].set_index("db_id").to_dict()["lang"]
     if "max_score" in data.columns:
          db_max_mean_similarity_df = (
               1 - data.groupby(["hardness", "lang"])["max_score"].mean()
          )
     if "avg_score" in data.columns:
          db_avg_mean_similarity_df = (
               1 - data.groupby(["hardness", "lang"])["avg_score"].mean()
          )
     _filter = data[label] == True
     true_label_count = (
          data[["hardness", "lang", label]][_filter].groupby(["hardness", "lang"]).count()
     )
     label_count = (
          data[["hardness", "lang", label]].groupby(["hardness", "lang"]).count()
     )
     db_id_df = (
          data[_filter].groupby(["hardness", "lang"]).count()
          / data.groupby(["hardness", "lang"]).count()
     )[[label]].fillna(0)
     db_id_df = db_id_df.rename(columns={label: "acc"})
     db_id_df["lang"] = db_id_df.index.map(db_lang_dict)
     db_id_df["true_count"] = true_label_count
     db_id_df["true_count"].fillna(0, inplace=True)
     db_id_df["count"] = label_count
     if "max_score" in data.columns:
          db_id_df["mean_avg_similarity"] = db_avg_mean_similarity_df
     if "avg_score" in data.columns:
          db_id_df["mean_max_similarity"] = db_max_mean_similarity_df
     db_id_df.drop(columns=["lang"], inplace=True)
     sorted_db_id_df = db_id_df.sort_values(["lang", "acc"], ascending=[False, False])
     sorted_db_id_df.reset_index(inplace=True)
     sorted_db_id_df["hardness-lang"] = (
          sorted_db_id_df["hardness"] + "-" + sorted_db_id_df["lang"]
     )
     return  sorted_db_id_df
def calculate_lang_label(label,data):
     _filter = data[label] == True
     true_label_count = data[["lang", label]][_filter].groupby("lang").count()
    
     label_count = data[["lang", label]].groupby("lang").count()
     lang_df = (
          (
               data[_filter].groupby(data["lang"]).count()
               / data.groupby(data["lang"]).count()
          )[[label]]
     ).fillna(0)
     lang_df["true_count"] = true_label_count
     lang_df["true_count"].fillna(0, inplace=True)
     lang_df["count"] = label_count
     lang_df = lang_df.rename(columns={label: "acc"})
     sorted_lang_df = lang_df.sort_values(["lang", "acc"], ascending=[False, False])
     sorted_lang_df.reset_index(inplace=True)
     return  sorted_lang_df

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
          "partial":round(len(res[res['partial_label'] == True]) / len(res)*100, 2) }
     


     # by lang
    
     hard_label=calculate_lang_label("hard_label",res)
     soft_label=calculate_lang_label("soft_label",res)
     partial_label=calculate_lang_label("partial_label",res)

     hard_en=hard_label[hard_label['lang']=='en']['acc']
     hard_en= round(hard_en.values[0]*100,2) if not hard_en.empty else None

     soft_en=soft_label[soft_label['lang']=='en']['acc']
     soft_en= round(soft_en.values[0]*100,2) if not soft_en.empty else None

     partial_en=partial_label[partial_label['lang']=='en']['acc']
     partial_en= round(partial_en.values[0]*100,2) if not partial_en.empty else None

     number_of_examples_en=hard_label[hard_label['lang']=='en']['count']
     number_of_examples_en= number_of_examples_en.values[0] if not number_of_examples_en.empty else 0.0



     hard_de=hard_label[hard_label['lang']=='de']['acc']
     hard_de= round(hard_de.values[0]*100,2) if not hard_de.empty else None

     soft_de=soft_label[soft_label['lang']=='de']['acc']
     soft_de= round(soft_de.values[0]*100,2) if not soft_de.empty else None

     partial_de=partial_label[partial_label['lang']=='de']['acc']
     partial_de= round(partial_de.values[0]*100,2) if not partial_de.empty else None

     number_of_examples_de=hard_label[hard_label['lang']=='de']['count']
     number_of_examples_de= number_of_examples_de.values[0] if not number_of_examples_de.empty else 0.0


     
     result_lang = {
          "en":
          {
               "hard":hard_en,
               "soft": soft_en,
               "partial": partial_en,
               "number_of_examples":int(number_of_examples_en)
          },
          "de":
           {
               "hard":hard_de,
               "soft": soft_de,
               "partial": partial_de,
               "number_of_examples":int(number_of_examples_de)
          },
     }
     
     hardness_hard_label=calculate_hardness_label("hard_label",res)
     soft_hard_label=calculate_hardness_label("soft_label",res)
     partial_hard_label=calculate_hardness_label("partial_label",res)

     
     hardness=['easy','medium', 'hard','extra','unknown']
     result_lang_hardness={lang:{} for lang in ['en','de']}
     for lang in ['en','de']:
               for hardness in ['easy','medium', 'hard','extra','unknown']:
                    hard= hardness_hard_label[ (hardness_hard_label['lang']==lang) & (hardness_hard_label['hardness']==hardness)]['acc']
                    hard= round(hard.values[0]*100,2) if not hard.empty else None

                    soft= soft_hard_label[ (soft_hard_label['lang']==lang) & (soft_hard_label['hardness']==hardness)]['acc']
                    soft= round(soft.values[0]*100,2) if not soft.empty else None

                    partial= partial_hard_label[ (partial_hard_label['lang']==lang) & (partial_hard_label['hardness']==hardness)]['acc']
                    partial= round(partial.values[0]*100,2) if not partial.empty else None
                    
                    number_of_examples=hardness_hard_label[(hardness_hard_label['lang']==lang) & (hardness_hard_label['hardness']==hardness)]['count']
                    number_of_examples=number_of_examples.values[0] if not number_of_examples.empty else 0.0

                    result_lang_hardness[lang][hardness]={
                              "hard":hard,
                              "soft":soft,
                              "partial":partial,
                              "number_of_examples":number_of_examples
                      }
               

    

     # by hardness

     print(result_lang_hardness)

     return result, result_lang, result_lang_hardness

# read the file with correct hardness
with open("data/statbot/nl2sql_data/final_dev_statbot_old.csv") as f:
     hardness_df = pd.read_csv(f, delimiter= ',')

hardness_df['question'] = hardness_df['question'].str.replace("\n","").str.strip()
print(len(hardness_df.index))



# look at  the csvs in results for each experiments:
results_path='data/statbot/gpt-3.5-turbo-16k/runs/'   #data/statbot/gpt-3.5-turbo-16k/runs/  data/statbot/mixtral/mulitlingual-embeddings
target="zero-shots"  #  random, mulitlingual-embeddings
shot = 0
csv_files = [f for f in os.listdir(os.path.join(results_path,target,"csv-outputs")) if  f.endswith('.csv')]
n_times_run_results_lang={i:[ ] for i in [shot]}
n_times_run_results_lang_hardness={i:[ ] for i in [shot]}
for csv_file in csv_files:
     print(csv_file)
     #shot_ = csv_file.split("-")[0]
     shot_=0
     if int(shot_)==shot:
          csv_file = os.path.join(results_path,target,"csv-outputs",csv_file)
          with open(csv_file) as f:
               d = pd.read_csv(f, delimiter= ',')
          for idx, row in hardness_df.iterrows():
               question = row['question']
               prediction = d.loc[d['question'] == question, 'generated_query'].values
               if len(prediction) > 0:
                    hardness_df.at[idx, 'generated_query'] = prediction[0]
          _ , result_lang,result_lang_hardness = calculate_ex_accuracy(hardness_df)
          n_times_run_results_lang[int(shot)].append(result_lang)
          n_times_run_results_lang_hardness[int(shot)].append(result_lang_hardness)

print(n_times_run_results_lang)
print(n_times_run_results_lang_hardness)

with open(f"{results_path}/{target}/{shot}-results_lang.json", "w") as output:
          json.dump(n_times_run_results_lang, output, indent=2, default=int)


with open(f"{results_path}/{target}/{shot}-results_lang.json") as input:
          d = json.load(input)
          results={l:{"hards":[],"softs":[],"partials":[],"number_of_examples":[]} for l in ['en','de'] }
          for shot, lang_rs in d.items():
               for r in lang_rs:
                    for lang, acc in r.items():
                         print(lang, acc)
                         hard=acc['hard']
                         soft=acc['soft']
                         partial=acc['partial']
                         number_of_examples=acc['number_of_examples']
                         results[lang]["hards"].append(float(hard))
                         results[lang]["softs"].append(float(soft))
                         results[lang]["partials"].append(float(partial))
                         results[lang]["number_of_examples"].append(float(number_of_examples))

          mean_std_lang={l:{} for l in ['en','de'] }
          for lang, rs in  results.items():

               hard_mean_var={
                    "mean":"%.2f" % np.mean(rs["hards"]),
                    "std":"%.2f" % np.std(rs["hards"])
                    
               }
               
               soft_mean_var={
                    "mean":"%.2f" % np.mean(rs["softs"]),
                    "std":"%.2f" % np.std(rs["softs"])
               }

               partial_mean_var={
                    "mean":"%.2f" % np.mean(rs["partials"]),
                    "std":"%.2f" % np.std(rs["partials"])
               }

               number_of_examples_mean_var={
                    "mean":"%.2f" % np.mean(rs["number_of_examples"]),
               }
               mean_std_lang[lang]={'hard':hard_mean_var,
                              'soft':soft_mean_var,
                              'partial':partial_mean_var,
                             'number_of_examples':number_of_examples_mean_var}

with open(f"{results_path}/{target}/{shot}_mean_std_lang.json","w") as llm:
               json.dump(mean_std_lang, llm, indent=2, default=int)
               

with open(f"{results_path}/{target}/{shot}-results_lang_hardness.json", "w") as output:
          json.dump(n_times_run_results_lang_hardness, output, indent=2, default=int)


with open(f"{results_path}/{target}/{shot}-results_lang_hardness.json") as input:
          d = json.load(input)
          results={l:{h:{"hards":[],"softs":[],"partials":[],"number_of_examples":[]} for h in ['easy','medium', 'hard','extra','unknown']} for l in ['en','de'] }
          for shot, lang_rs in d.items():
               for x in lang_rs:
                    for lang, r in x.items():
                         hards=[]
                         softs=[]
                         partials=[]
                         n_examples=[]
                         for h,acc in r.items(): 
                              hard=acc['hard']
                              soft=acc['soft']
                              partial=acc['partial']
                           
                              n_example=acc['number_of_examples']
                              if hard is not None:
                                   results[lang][h]["hards"].append(float(hard))
                              else:
                                   results[lang][h]["hards"].append(0)
                              
                              if soft is not None:
                                   results[lang][h]["softs"].append(float(soft))
                              else:
                                   results[lang][h]["softs"].append(0)

                              if partial is not None:
                                   results[lang][h]["partials"].append(float(partial))
                              else:
                                   results[lang][h]["partials"].append(0)
                              
                              results[lang][h]["number_of_examples"].append(n_example)
     
          results_st_mean={l: {h:{} or h in ['easy','medium', 'hard','extra','unknown'] } for l in ['en','de'] }
          
          for l, h in results.items():
            
               for hardness, acc in h.items():

                    hard_mean_var={
                         "mean":"%.2f" % np.mean(h[hardness]["hards"]),
                         "std":"%.2f" % np.std(h[hardness]["hards"])
                         
                    }
                              
                    soft_mean_var={
                         "mean":"%.2f" % np.mean(h[hardness]["softs"]),
                         "std":"%.2f" % np.std(h[hardness]["softs"])
                    }

                    partial_mean_var={
                         "mean":"%.2f" % np.mean(h[hardness]["partials"]),
                         "std":"%.2f" % np.std(h[hardness]["partials"])
                    }

                    examples_mean={
                         "mean":"%.2f" % np.mean(h[hardness]["number_of_examples"]),
                         "std":"%.2f" % np.std(h[hardness]["number_of_examples"])
                    }
                    results_st_mean[l][hardness]={'hard':hard_mean_var,
                                   'soft':soft_mean_var,
                                   'partial':partial_mean_var,
                                   "examples":examples_mean}
               
with open(f"{results_path}/{target}/{shot}_mean_std_lang_hardness.json","w") as llm:
               json.dump(results_st_mean, llm, indent=2, default=int)