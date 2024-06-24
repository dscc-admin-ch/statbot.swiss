import csv, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from langchain.prompts import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd
from sqlalchemyWrapper import schema_db_postgres_statbot
from sql_analytics.src.utils.exec_eval import *
from sql_analytics.src.utils.sql_database import SQLDatabase
import torch
from torch import cuda, bfloat16
import time
import random
import json
import numpy as np
import argparse
from utility import *

def test_sqls(host, port, database, username, password, data):
    database_uri = f"postgresql://{username}:{password}@{host}:{str(port)}/{database}"
    sqldb = SQLDatabase.from_uri(
        database_uri,
        engine_args={"connect_args": {"options": "-c statement_timeout=120000"}},
    )
    sqldb._schema = "experiment"
    tbl = []
    for d in tqdm(data):  # data is a record
        _db_id = d["db_id"]
        _question = d["question"]
        _gt = d["generated_query"]
        _pred = d["query"]
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
def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'[Question]: {message} [/INST]')
    return ''.join(texts)


def zero_shot_template():
    prifix='''<s>[INST] <<SYS>>You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}\n'''
    zero_shot_prompt_template = PromptTemplate(
            input_variables=["input","table_info"],
            template=prifix+"<</SYS>>\n### Question\n{input}\n### SQL query[/INST]\n",
    )
    return zero_shot_prompt_template




def few_shot_template(example_prompt,example_selector):
    similar_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = '''<s>[INST] <<SYS>>\nGiven an input question, Please create only a syntactically correct {dialect} SQL query. 
Never query for all the columns from a specific table, only ask for the few relevant columns given the question. 
Pay attention to using only the column names that you can see in the schema description. 
Be careful to not query for columns that do not exist. 
Also, pay attention to which column is in which table. If more than one table participates use the JOIN.
Use the provided tables, columns, foreign keys, and primary keys to generate the Postgres SQL query.
Use the database values that are explicitly mentioned in the question.
Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
Use DESC and DISTINCT when needed.
Pay attention to the columns that are used for the GROUP BY statement.
Pay attention to the columns that are used for the SELECT statement.
Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
        
Only use the tables listed in the database schema information.

[Database Schema]:\n\n{table_info}.

[Examples]:\n\n
''',
        suffix="\n<</SYS>>Generate the SQL query for the following question, considering the above examples: [Question]: {question}[/INST]\n",
        input_variables=["question","table_info","dialect"]
        )
    

    return similar_prompt


def zero_shot_template_mistral():
    prifix='''[INST]You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}'''

    zero_shot_prompt_template = PromptTemplate(
            input_variables=["input","table_info"],
            template=prifix+"### Question\n{input}\n### SQL query[/INST]\n",
    )
    return zero_shot_prompt_template


def few_shot_template_examples_mistral(example_prompt,example_selector):
    # suffix=''
    # for nl2slq in nl2sql_pairs:
    #     suffix+= f'[Question]: {nl2slq[0]}\n[SQL Query]: {nl2slq[-1]}\n\n'
    # suffix='Please include the following exampes for better understanding.\n\n[Examples]\n\n'+suffix
    similar_prompt_rules = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = '''[INST] You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}''',
        suffix="\n### Question\n{input}\n#### SQL query [/INST]\n",
        input_variables=["input","table_info"]
        )

    return similar_prompt_rules


def few_shot_template_examples_mistral_random(example_prompt,examples):
    # suffix=''
    # for nl2slq in nl2sql_pairs:
    #     suffix+= f'[Question]: {nl2slq[0]}\n[SQL Query]: {nl2slq[-1]}\n\n'
    # suffix='Please include the following exampes for better understanding.\n\n[Examples]\n\n'+suffix
    similar_prompt_rules = FewShotPromptTemplate(
        examples= examples,
        example_prompt = example_prompt,
        prefix = '''[INST] You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}''',
        suffix="\n### Question\n{input}\n#### SQL query [/INST]\n",
        input_variables=["input","table_info"]
        )

    return similar_prompt_rules

def extract_shots(examples,question,embeddings, max_shots):
    ####RULES###
    few_shot_examples=[]
    meta_data=[]
    for j in range(len(examples)):
        #question,query,
        ex_question = examples.loc[j,'question'].replace("\n","").strip()
        ex_query = examples.loc[j,'query']
        # ex_rules=examples.loc[j,'rules']
        few_shot_examples.append({"question": ex_question})
        # meta_data.append({"question":ex_question,"rules":"\n".join(ex_rules.split(", ")),"query":ex_query})
        meta_data.append({"question":ex_question,"query":ex_query})

    to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
    # ids=[str(i) for i in range(1, len(few_shot_examples) + 1)]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=meta_data)

    # Lower score is more similar
    answers = vectorstore.similarity_search_with_score(question, max_shots)
    nl2sql_pairs=[]
    for item in answers:
        #print(item[0].metadata['question']) # print out score          
        # print(item[0].metadata['query']) 
        # print(item[0].metadata['rules']) 
        nl = item[0].metadata['question']
        q = item[0].metadata['query']
        score = item[1]
        nl2sql_pairs.append(
            {'question':nl, 'query':q, 'score':score}
        )
    scores=[item[1] for item in answers]

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k = max_shots,
    )

    # for examples
    examples_prompt = PromptTemplate(
    input_variables=["question","query"],
    template="### Question\n{question}\n### SQL query\n{query}\n",
    )  

    # # for rules and examples
    # examples_prompt = PromptTemplate(
    #     input_variables=["rules"],
    #     template="{rules}",
    # )
    system_prompt = few_shot_template_examples_mistral(examples_prompt, example_selector)
    return system_prompt, scores, vectorstore, nl2sql_pairs


def extract_shots_randomly(examples,question,embeddings, max_shots):
    random.seed(42)
    ####RULES###
    few_shot_examples=[]
    meta_data=[]
    for j in range(len(examples)):
        #question,query,
        ex_question = examples.loc[j,'question'].replace("\n","").strip()
        ex_query = examples.loc[j,'query']
        # ex_rules=examples.loc[j,'rules']
      
        # meta_data.append({"question":ex_question,"rules":"\n".join(ex_rules.split(", ")),"query":ex_query})
        few_shot_examples.append({"question":ex_question,"query":ex_query})


    nl2sql_pairs=[]
    scores=[]
    for item in random.choices(few_shot_examples, k=min(max_shots,len(few_shot_examples))):
        #print(item[0].metadata['question']) # print out score          
        # print(item[0].metadata['query']) 
        # print(item[0].metadata['rules']) 
        score = 0
        item['score']=score
        nl2sql_pairs.append(item)
        scores.append(score)
    
   
    # for examples
    examples_prompt = PromptTemplate(
    input_variables=["question","query"],
    template="### Question\n{question}\n### SQL query\n{query}\n",
    )  

    # # for rules and examples
    # examples_prompt = PromptTemplate(
    #     input_variables=["rules"],
    #     template="{rules}",
    # )
    system_prompt = few_shot_template_examples_mistral_random(examples_prompt, nl2sql_pairs)
    return system_prompt, scores,None, nl2sql_pairs

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data-path", type=str, default="/workspace/dataset/statbot", 
                       help="path to the statbot folder")
    parse.add_argument("--shot", type=str,
                       help="zero_shot or few_shot", default='few-shot', required=False)
    # model parameters
    parse.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1") #, meta-llama/Llama-2-7b-chat-hf/ mistralai/Mistral-7B-Instruct-v0.2"/ meta-llama/Llama-2-13b-chat-hf, mistralai/Mixtral-8x7B-Instruct-v0.1
    parse.add_argument("--temperature", type=float, default= 0)
    parse.add_argument("--sample-rows", type=int, default= 5)
    parse.add_argument("--fraction", type=float, default= 5)
    parse.add_argument("--random-seed", type=int, default = 42)

    # outputs
    parse.add_argument("--results-file-dir", type=str, default="/workspace/dataset/statbot/outputs")
    return parse.parse_args()




def main():
    random.seed(42)
    args = get_args()
    data_path=args.data_path
    random.seed(args.random_seed)
    print("GPU:",torch.cuda.is_available())
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # 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
    model_name = args.model_name # <-- You can switch to other models like "NumbersStation/nsql-6B" # NumbersStation/nsql-llama-2-7B
    version = model_name.split("/")[-1]

    shot=args.shot
    
    tic = time.perf_counter()

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    hf_auth='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'
    # model_config=AutoConfig.from_pretrained(model_name,use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth)
    model_config = AutoConfig.from_pretrained(
        model_name,
        token=hf_auth
    )
    max_memory = f'{40960}MB'
    n_gpus = torch.cuda.device_count()
    print(n_gpus)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=True,
        #quantization_config=bnb_config ,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        token=hf_auth,
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    model.eval()
    ###
    max_token_length= 4096 if 'mistral' not in model_name else 7000
   
    
    print(f"Model loaded on {device}")
    toc = time.perf_counter()
    process_time=toc-tic
    print(f"Process Time= {process_time:0.4f} second")
    print(f"Model loaded on {device}")

    ### split the dataset to dev (30%) and test(70%) for few-shot based on each tabel:
    with open(f"{data_path}/nl2sql_data/test-old.csv") as f:
        d = pd.read_csv(f, delimiter= ',')

    ## rules
   
    with open(f"{args.data_path}/nl2sql_data/nl2sql2qdmr.csv") as f: 
        shots_with_explanations = pd.read_csv(f, delimiter= ',')

    # with open(f"{args.data_path}/nl2sql_data/explanations_gpt_old.csv") as f: 
    #     shots_with_explanations = pd.read_csv(f, delimiter= ',')
    #     #'dataset/statbot/nl2sql_data/zero_shot_rules_gpt_old.csv'


    with open(f"{args.data_path}/nl2sql_data/train-old.csv") as f:
        origin_of_shots = pd.read_csv(f, delimiter= ',')

    # origin_of_shots.rename(columns = {'db_id_x':'db_id'}, inplace = True)
    shots_with_explanations['question'] =  shots_with_explanations['question'].str.strip()
    origin_of_shots['question'] = origin_of_shots['question'].str.strip()

    intersection_arr = np.intersect1d(shots_with_explanations['question'], origin_of_shots['question'])
    shots_with_explanations = shots_with_explanations.loc[shots_with_explanations['question'].isin(intersection_arr),:]


    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
    embeddings = HuggingFaceEmbeddings()
    
    
   
    for number_of_shots in [1,2,3,4,5,6,8]:
        llm_results=[]
        for run in range(3):
            all_results=[]
            ###
            # ####
            # if (os.path.exists(results_file)==False):
            #     f = open(results_file,"w")
            #     json.dump([], f)
            # else:
            #     f = open(results_file, "w+")
            #     json.dump([], f)


            # with open(results_file, "r+") as output:
            i=0
            
            # for index, e in d.iterrows():
            vectorstore = None
            for idx in tqdm(range(len(d.index))):
                max_shots = int(number_of_shots)
                question = d.loc[idx, 'question'].replace("\n","").strip()
                query = d.loc[idx, 'query']
                db_id = d.loc[idx, 'db_id']
                lang = d.loc[idx,'lang']
                hardness = d.loc[idx,'hardness']
                print("question",question)
                print("#######################")
                db_schema = schema_db_postgres_statbot( include_tables=['spatial_unit', db_id], sample_number=args.sample_rows)
                if vectorstore is not None:
                    ########CLEAR THE VECTORSTORE
                    vectorstore.delete_collection()
                # chat_history=[]
                if number_of_shots==0:
                    system_prompt= zero_shot_template_mistral()
                    scores=[]
                    nl2sql_pairs=[]
                else:
                    ####RULES###
                    examples = shots_with_explanations.loc[shots_with_explanations['db_id']==db_id]
                    examples = examples.drop(examples[examples["question"] == question.strip()].index)
                    examples=examples.reset_index()
                    
                    # system_prompt,scores,vectorstore,nl2sql_pairs=extract_shots_with_explanation(examples=examples,
                    #                                                             question=question,
                    #                                                             embeddings=embeddings,
                    #                                                             max_shots=max_shots)
                    
                    system_prompt,scores,vectorstore,nl2sql_pairs = extract_shots_randomly(examples=examples,
                                                                                question=question,
                                                                                embeddings=embeddings,
                                                                                max_shots=max_shots)
                        
                # input_nl2sql = get_prompt(question, chat_history=chat_history, system_prompt=system_prompt)
                input_nl2sql=system_prompt.format(input=question,table_info=db_schema)
                print(input_nl2sql)
                # vectorstore.delete_collection()

                length = tokenizer([input_nl2sql], return_tensors='np', add_special_tokens=False)['input_ids'].shape[-1]
                print(f'###################### Generation for item:{i} with length {length}####################################')

                number_or_iteration=0
                while length > max_token_length:
                    if number_or_iteration==args.sample_rows:
                        break
                    if number_of_shots==0:
                        sample_rows = args.sample_rows-1
                        db_schema = schema_db_postgres_statbot( include_tables=['spatial_unit', db_id], sample_number=sample_rows)
                        input_nl2sql=system_prompt.format(input=question,table_info=db_schema)
                        print(input_nl2sql)
                        length=tokenizer([input_nl2sql], return_tensors='np', add_special_tokens=False)['input_ids'].shape[-1]

                    else:    
                        sample_rows = args.sample_rows-1
                        db_schema = schema_db_postgres_statbot( include_tables=['spatial_unit', db_id], sample_number=sample_rows)
                        max_shots = max_shots -1 
                        max_shots=max_shots if max_shots >0 else 1
                        system_prompt,scores,vectorstore, nl2sql_pairs = extract_shots_randomly(examples=examples,
                                                                                    question=question,
                                                                                    embeddings=embeddings,
                                                                                    max_shots=max_shots)
                        input_nl2sql=system_prompt.format(input=question,table_info=db_schema)
                        print(input_nl2sql)
                        length=tokenizer([input_nl2sql], return_tensors='np', add_special_tokens=False)['input_ids'].shape[-1]
                    number_or_iteration+=1

                if number_or_iteration==args.sample_rows:
                        llm_output=""
                        SQL="SELECT"
                        process_time=0

                else:

                    tic = time.perf_counter()
                    input_ids = tokenizer(input_nl2sql, return_tensors="pt").input_ids
                    input_ids = input_ids.to('cuda')
                    params={
                    "temperature":0,
                    "do_sample":False,
                    "num_beams":1,
                    "num_return_sequences":1,
                    "top_p":1.0,
                    "max_new_tokens":500,
                    "pad_token_id":tokenizer.eos_token_id
                    }
                    generated_ids = model.generate(input_ids,**params)
                    llm_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    llm_output = llm_output.split('[/INST]')[-1]
                    print(llm_output.replace("\\",""))
                    toc = time.perf_counter()
                    process_time = toc-tic
                    print(f"Process Time for inference= {process_time:0.4f} second")
                    #### Run the SQL Statement get feedback 
                    SQL=extractSQLpart(llm_output).replace("SELECT",'').replace("\\","")
                    SQL=f'SELECT {SQL}'
                    print(f"Generated SQL: {SQL}")

                result = {
                    "i":i,
                    "question": question,
                    "query":query,
                    "db_id":db_id,
                    "llm_output":llm_output.replace("\n"," ").replace("\n\n"," ").replace("  "," ").replace("\\",""),
                    "generated_query":SQL.replace("\n"," ").replace("\n\n"," ").replace("  "," ").replace("\\",""),
                    # "few_shots":"NA" if 'zero' in shot else example_selector,
                    "prompt": input_nl2sql,
                    "hardness":hardness,
                    "lang":lang,
                    "process_time":process_time,
                    "input_length":length,
                    "scores": scores,
                    "meta_data":nl2sql_pairs,
                }

                all_results.append(result)
                i += 1

                # output.seek(0)
            results_file = f'{args.results_file_dir}/{version}-{number_of_shots}-shot-train-test-old-random-run#{run}.json'
            with open(results_file,"w") as output:
                json.dump(all_results, output, indent=2)
            
            with open(results_file) as f:
                all_results = json.load(f)
            
            #csv_output=f'{data_path}/outputs/{number_of_shots}-shot_{version}-train-test-old.csv'
            csv_output=f'{data_path}/outputs/{number_of_shots}-shot_{version}-train-test-old-random-run#{run}.csv'

            with open(csv_output,'w',newline='',encoding='utf8') as file:
                writer = csv.writer(file)
                writer.writerow(['db_id',"question","query","generated_query","hardness","lang","process_time","input_token_length","Max_Score","Avg_Score"])

                for item in all_results:
                    Max_Score=0 if item['scores']== [] else  min(item['scores']) 
                    Avg_Score=0 if item['scores']==[ ] else sum(item['scores'])/len(item['scores'])
                    writer.writerow([item['db_id'],item["question"],item["query"].strip(), item['generated_query'].strip(),item['hardness'].strip(),item['lang'].strip(),item['process_time'],item['input_length'],Max_Score,Avg_Score])
            
            data = pd.read_csv(csv_output)
            data.generated_query = data.generated_query.fillna('SELECT')
            data=data.to_dict("records")
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
            print(csv_output)

        
            
            result={
                "experiment":number_of_shots,
                "model":version,
                "hard":round(len(res[res['hard_label'] == True]) / len(res)*100, 2),
                "soft":round(len(res[res['soft_label'] == True]) / len(res)*100, 2),
                "partial":round(len(res[res['partial_label'] == True]) / len(res)*100, 2)
            }
        
            llm_results.append(result)
            csv_output_performance=f"/workspace/dataset/statbot/llm_results/{number_of_shots}-shot_{version}-random-run#{run}.csv"

            with open(csv_output_performance,'w',newline='',encoding='utf8') as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow([result['hard'],result['soft'],result['partial']])

        
        with open(f'/workspace/dataset/statbot/llm_results/mixtral7x8b-{number_of_shots}-shot-random.json',"w") as llm:
                json.dump(llm_results, llm, indent=2)
        
if __name__ == '__main__':
    main()