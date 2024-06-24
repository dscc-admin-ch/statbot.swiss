import sys
import time
import random
import json
import copy
import argparse
import re,os
import pandas as pd
from tqdm import tqdm

import numpy  as np
#sys.path.insert(0, "./langchain")
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain

from langchain.chains.sql_database.prompt import DECIDER_PROMPT
from langchain.prompts import load_prompt
from langchain.chains.llm import LLMChain
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

import tiktoken
random.seed(123)
from generation import *



from sql_analytics.src.utils.exec_eval import *
from sql_analytics.src.utils.sql_database import SQLDatabase

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
        _Max_Score = d["Max_Score"]
        _Avg_Score = d["Avg_Score"]
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
                "max_score": _Max_Score,
                "avg_score": _Avg_Score,
            }
        )
    return pd.DataFrame(tbl)

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data-path", type=str, default="/data/statbot",
                       help="path to the data folder")
    parse.add_argument("--shot-selection-strategy", type=str, default='similarity', help="what is the strategy for selecting the examples as a shot")

    # model parameters
    parse.add_argument("--temperature", type=float, default=0)
    parse.add_argument("--model-name", type=str, default="gpt-3.5-turbo-16k", help="form openai code-davinci-002,gpt-3.5-turbo, gpt-3.5-turbo-16k ")
    parse.add_argument("--sample-rows", type=int, default=5)

    # outputs
    return parse.parse_args()




def main():

    random.seed(42)
    args = get_args()
    # parameters read from config.json
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    print(config)
    os.environ["OPENAI_API_KEY"] = config['open_api_key']
    db_config=config['db']
    host = db_config['host']
    port = db_config['port']
    database = db_config['database']
    username = db_config['username']
    password = db_config['password']
    schema = db_config['schema']
    database_uri = f"postgresql://{username}:{password}@{host}:{str(port)}/{database}"
    sqldb = SQLDatabase.from_uri(database_uri)
    sqldb._schema = "experiment"

    llm = OpenAI(temperature = args.temperature,
                 model_name = args.model_name,
                 n = 1,
                 stream = False,
                 max_tokens = 500,
                 top_p = 1.0,
                 frequency_penalty = 0.0,
                 presence_penalty = 0.0,
                 stop=";\n"
                 )

    # creating  the embedding for calculate the similarity during the shot selection
    ## english embeddings
    # embeddings = HuggingFaceEmbeddings()
    ## multilingual embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
   
    n_shots=[0,1,2,3,4,5,6,8]
    n_iterations = 3
    ################################################################################################
    for shot in n_shots:
        n_times_run_results=[]
        for i in range(n_iterations):
            if shot==0:
                data_file= generate_sql_zero_shots(args,db_config,llm,i)
            else:
                if args.shot_selection_strategy =='random':
                
                    data_file= generate_sql_in_context_learning_random_shots(args,db_config,llm,shot,i)
                elif args.shot_selection_strategy =='similarity':
                    data_file = generate_sql_in_context_learning_similar_shots(args,db_config,llm, embeddings,shot,i)
                else:
                    print("select a value fselection process!")
        # #################################evaluation#########################
            print(data_file)
            data = pd.read_csv(data_file).to_dict("records")

            res = test_sqls(host, port, database, username, password, data)
            
            print(
            f"hard: {round(len(res[res['hard_label'] == True]) / len(res)*100, 2)}%\nsoft: {round(len(res[res['soft_label'] == True]) / len(res)*100, 2)}%\npartial: {round(len(res[res['partial_label'] == True]) / len(res)*100, 2)}%"
            )
            
            result = {"hard":round(len(res[res['hard_label'] == True]) / len(res)*100, 2),
                    "soft":round(len(res[res['soft_label'] == True]) / len(res)*100, 2),
                    "partial":round(len(res[res['partial_label'] == True]) / len(res)*100, 2) }
        
            n_times_run_results.append(result)
            with open(f"data/statbot/gpt-3.5-turbo-16k/runs/{shot}-run-{i}-random.json", "w") as output:
                json.dump(result, output, indent=2)

        with open(f"data/statbot/gpt-3.5-turbo-16k/runs/{shot}-random-run-{i}", "w") as output:
                json.dump(n_times_run_results, output, indent=2)



if __name__ == '__main__':
    main()

