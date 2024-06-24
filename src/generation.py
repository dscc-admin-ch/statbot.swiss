import sys
import time
import random
import json
import copy
import argparse
import re
import pandas as pd
from tqdm import tqdm

import numpy  as np
#sys.path.insert(0, "./langchain")

from langchain.chains.sql_database.prompt import DECIDER_PROMPT
from langchain.prompts import load_prompt
from langchain.chains.llm import LLMChain
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import FAISS



import tiktoken
# random.seed(123)
from sqlalchemyWrapper import schema_db_postgres_statbot
import os

import csv

def num_tokens_from_string(string: str, encoding_name:str) -> int:
                """Returns the number of tokens in a text string."""
                encoding = tiktoken.encoding_for_model(encoding_name)
                num_tokens = len(encoding.encode(str(string)))
                return num_tokens

def zero_shot_template():
  
    prompt='''You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the SQL query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}
   
    ### Question: 
    {input}
    ### SQL query:\n
'''
    zero_shot_prompt_template = PromptTemplate(
        input_variables=["input","table_info"],
        template=prompt,
    )
    return zero_shot_prompt_template

def few_shot_template_examples(example_prompt, example_selector):
    
  
    prefix = '''You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the SQL query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}
'''

    few_shot_prompt = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        example_selector = example_selector,
        example_prompt = example_prompt,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix= prefix,
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix= "### Question\n{input}\n### SQL query\n",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["input", "table_info"],
        example_separator="\n\n",
    )
    return few_shot_prompt

def few_shot_template_examples_random(example_prompt, examples):
    
  
    prefix = '''You are an helpful AI assistant who writes SQL query for a given question. Given the database described by the database schema below, write a SQL query that answers the question.\nDo not explain the SQL query.\nReturn just the query, so it can be run verbatim from your response.\n### Database Schema\n{table_info}
'''

    few_shot_prompt = FewShotPromptTemplate(
        # These are the examples we want to insert into the prompt.
        examples = examples,
        example_prompt = example_prompt,
        # The prefix is some text that goes before the examples in the prompt.
        # Usually, this consists of intructions.
        prefix= prefix,
        # The suffix is some text that goes after the examples in the prompt.
        # Usually, this is where the user input will go
        suffix= "### Question\n{input}\n### SQL query\n",
        # The input variables are the variables that the overall prompt expects.
        input_variables=["input", "table_info"],
        example_separator="\n\n",
    )
    return few_shot_prompt





def generate_sql_zero_shots(args,db_config,llm,i):
    results_file = f"{args.data_path}/{args.model_name}/zero-shot_run#{i}.json"
                    
    if (os.path.exists(results_file) == False):
        with open(results_file, "w") as f:
            json.dump([], f)
    else:
        with open(results_file, "w+") as f:
            json.dump([], f)
    

    print(f"-----------Reading the file: {results_file}---------")

    with open(f"{args.data_path}/nl2sql_data/test.csv") as f:
        d = pd.read_csv(f, delimiter= ',')

   
    with open(results_file, "r+") as output:
        all_results = json.load(output)
        iddx = 0
        print(len(d.index))
        for iddx in tqdm(range(len(d.index))):
            #question,query,prediction,spider_hardness,gt_res,pred_res,label
            question = d.loc[iddx,'question'].replace("\n","").strip()
            query = d.loc[iddx,'query']
            hardness=d.loc[iddx,'hardness']
            db_id = d.loc[iddx,'db_id']
            lang = d.loc[iddx,'lang']
      
        
            prompt_template = zero_shot_template()
            llm_chain= LLMChain(llm=llm, prompt= prompt_template)
                
            sql = None
            
            ddl = schema_db_postgres_statbot(db_config,include_tables=['spatial_unit', db_id],lang=lang,
                                    sample_number=args.sample_rows)
            llm_inputs = {
                "input": question,
                "table_info": ddl,
            }

            prompts = llm_chain.prep_prompts([llm_inputs])
            
            prompt_strings = [p for p in prompts[0]][0].to_string() 
            print(prompt_strings)
            # check the length:
            # Write function to take string input and return number of tokens
            num_tokens = num_tokens_from_string(prompt_strings, args.model_name)

            print(f"Starting  generation: #input-tokens {num_tokens}")
            tic =time.perf_counter()
            while sql is None:
                try:
                    sql = llm_chain.run(**llm_inputs)
                    print(sql)
                except Exception as e:
                    print(str(e))
                    time.sleep(3)
                    pass
            toc = time.perf_counter()
            
            process_time=toc-tic
            print(f"Process Time= {process_time:0.4f} second")
            print('##############################')                    
            r = {
                "i": iddx,
                "question": question,
                "query": query,
                "db_id" : db_id,
                "generated_query":sql,
                "prompt": str(prompt_strings),
                "process_time":process_time,
                "num_tokens":num_tokens,
                "hardness":hardness,
                "lang":lang
                }

            all_results.append(r)
            iddx += 1

            # Sets file's current position at offset.
            output.seek(0)
            json.dump(all_results, output, indent=2)

    with open(results_file) as f:
        all_results = json.load(f)
    
    with open(f'{args.data_path}/outputs/gpt-3.5-turbo-with-zero-shot-examples-run{i}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['db_id',"question","query","generated_query","hardness","lang","process_time","Max_Score","Avg_Score"])
        for item in all_results:
            Max_Score= 0
            Avg_Score= 0
            writer.writerow([item['db_id'],item["question"],item["query"].strip(), item['generated_query'].strip(),item['hardness'].strip(),item['lang'].strip(),item['process_time'],Max_Score,Avg_Score])


    return file.name


def generate_sql_in_context_learning_random_shots(args,db_config,llm, max_shot,run):
    random.seed(42)
    results_file = f"{args.data_path}/{args.model_name}/runs/{max_shot}-shot_with_examples-random_train_70-30_test_old_run#{run}.json"
                    
    if (os.path.exists(results_file) == False):
        with open(results_file, "w") as f:
            json.dump([], f)
    else:
        with open(results_file, "w+") as f:
            json.dump([], f)
    
    with open (f"{args.data_path}/{args.model_name}/zero_shot_rules_from_gpt_failures-old-schema_extraction.json") as r:
        rules_db_ids = json.load(r)


    print(f"-----------Reading the file: {results_file}---------")

    with open(f"{args.data_path}/nl2sql_data/test.csv") as f:
        d = pd.read_csv(f, delimiter= ',')

    with open(f"{args.data_path}/nl2sql_data/train.csv") as f:
        origin_of_shots = pd.read_csv(f, delimiter= ',')
    
   
    with open(results_file, "r+") as output:
        all_results = json.load(output)
        
        print(len(d.index))
       
        for iddx in tqdm(range(len(d.index))): #len(d.index)
        #for iddx in tqdm(range(1)): #len(d.index)
            #question,query,prediction,spider_hardness,gt_res,pred_res,label
            question = d.loc[iddx,'question'].replace("\n","").strip()
            query = d.loc[iddx,'query']
            hardness=d.loc[iddx,'hardness']
            db_id = d.loc[iddx,'db_id']
            lang = d.loc[iddx,'lang']
            
            # if db_id!='criminal_offences_registered_by_police':
            #     continue

            ddl = schema_db_postgres_statbot(db_config,include_tables=['spatial_unit',db_id],lang=lang,
                                    sample_number=args.sample_rows)
        
            ####RULES###
            examples = origin_of_shots.loc[origin_of_shots['db_id']==db_id]
            examples = examples.drop(examples[examples["question"] == question.strip()].index)
            examples = examples.reset_index()
            few_shot_examples = []
            meta_data = []
            
            for j in range(len(examples)):
                #question,query,
                ex_question = examples.loc[j,'question'].replace("\n","").strip()
                ex_query = examples.loc[j,'query']
                few_shot_examples.append({"question":ex_question,"query":ex_query})
            
            
            
            # print (f'######{question}###########')
            nl2sql_pairs=[]
            scores=[]
            for item in random.choices(few_shot_examples, k=max(max_shot,len(few_shot_examples))):
                nl2sql_pairs.append(
                    item
                )
                scores.append(0)
            # print('###########') 
           
            examples_prompt = PromptTemplate(
                input_variables=["question","query"],
                template="### Question\n{question}\n### SQL query\n{query}",
            )  
            prompt_template = few_shot_template_examples_random(examples_prompt, nl2sql_pairs)
            llm_chain= LLMChain(llm=llm, prompt= prompt_template)
            
            sql = None
            
            llm_inputs = {
                "input": question,
                "table_info": ddl,
            }

            prompts = llm_chain.prep_prompts([llm_inputs])
           
            prompt_strings = [p for p in prompts[0]][0].to_string() 
            print(prompt_strings)
            # check the length:
            # Write function to take string input and return number of tokens
            num_tokens = num_tokens_from_string(prompt_strings, args.model_name)

            print(f"Starting  generation item {iddx} #input-tokens {num_tokens}")
            tic =time.perf_counter()
            while sql is None:
                try:
                    sql = llm_chain.run(**llm_inputs)
                    print(sql)
                except Exception as e:
                    print(str(e))
                    time.sleep(3)
                    pass
            toc = time.perf_counter()
            
            process_time=toc-tic
            print(f"Process Time= {process_time:0.4f} second")
            print('##############################')                    
            r = {
                "i": iddx,
                "question": question,
                "query": query,
                "db_id" : db_id,
                "generated_query":sql,
                "meta_data":nl2sql_pairs,
                "prompt": str(prompt_strings),
                "process_time":process_time,
                "num_tokens":num_tokens,
                "hardness":hardness,
                "scores": scores,
                "lang":lang
                }

            all_results.append(r)
            

            # Sets file's current position at offset.
            output.seek(0)
            json.dump(all_results, output, indent=2)

    with open(results_file) as f:
        all_results = json.load(f)
    
    with open(f'{args.data_path}/outputs/{args.model_name}/{max_shot}-shot-examples-random-train-70-30-test_old_run#{run}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['db_id',"question","query","generated_query","hardness","lang","process_time","Max_Score","Avg_Score"])
        for item in all_results:
            Max_Score= min(item['scores'])
            Avg_Score= sum(item['scores'])/len(item['scores'])
            writer.writerow([item['db_id'],item["question"],item["query"].strip(), item['generated_query'].strip(),item['hardness'].strip(),item['lang'].strip(),item['process_time'],Max_Score,Avg_Score])

    return file.name


def generate_sql_in_context_learning_similar_shots(args,db_config,llm, embeddings, max_shot,run):
    results_file = f"{args.data_path}/{args.model_name}/runs/{max_shot}-shot_with_examples-multil_train_70-30_test_old_en_emb_run#{run}.json"
                    
    if (os.path.exists(results_file) == False):
        with open(results_file, "w") as f:
            json.dump([], f)
    else:
        with open(results_file, "w+") as f:
            json.dump([], f)
    
    with open (f"{args.data_path}/{args.model_name}/zero_shot_rules_from_gpt_failures-old-schema_extraction.json") as r:
        rules_db_ids = json.load(r)


    print(f"-----------Reading the file: {results_file}---------")

    with open(f"{args.data_path}/nl2sql_data/test.csv") as f:
        d = pd.read_csv(f, delimiter= ',')

    with open(f"{args.data_path}/nl2sql_data/train.csv") as f:
        origin_of_shots = pd.read_csv(f, delimiter= ',')
    
   
    with open(results_file, "r+") as output:
        all_results = json.load(output)
        
        print(len(d.index))
        # for item in d[:10]:
        #["db_id","question","query","generated_query","errors","rules","label","hardness"])
        vectorstore=None
        for iddx in tqdm(range(len(d.index))): #len(d.index)
        #for iddx in tqdm(range(1)): #len(d.index)
            #question,query,prediction,spider_hardness,gt_res,pred_res,label
            
            question = d.loc[iddx,'question'].replace("\n","").strip()
            # if question !="What was the proportion of electric vehicles in Geneva in 2010?":
            #     continue
            query = d.loc[iddx,'query']
            hardness=d.loc[iddx,'hardness']
            db_id = d.loc[iddx,'db_id']
            lang = d.loc[iddx,'lang']
            
            # if db_id!='criminal_offences_registered_by_police':
            #     continue

            ddl = schema_db_postgres_statbot(db_config,include_tables=['spatial_unit',db_id],lang=lang,
                                    sample_number=args.sample_rows)
            
            if vectorstore is not None:
                ########CLEAR THE VECTORSTORE
                vectorstore.delete_collection()
            ####RULES###
            examples = origin_of_shots.loc[origin_of_shots['db_id']==db_id]
            examples = examples.drop(examples[examples["question"] == question.strip()].index)
            examples = examples.reset_index()
            few_shot_examples = []
            meta_data = []
            
            for j in range(len(examples)):
                #question,query,
                ex_question = examples.loc[j,'question'].replace("\n","").strip()
                ex_query = examples.loc[j,'query']
                few_shot_examples.append({"question":ex_question})
                meta_data.append({"question":ex_question,"query":ex_query})
            
            to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
            # ids=[str(i) for i in range(1, len(few_shot_examples) + 1)]
            vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=meta_data)
            
            # Lower score is more similar
            answers = vectorstore.similarity_search_with_score(question, max_shot)

            # print (f'######{question}###########')
            nl2sql_pairs=[]
            scores=[]
            for item in answers:
                print(item[0].metadata['question']) # print out score          
                # print(item[0].metadata['query']) 
                # print(item[0].metadata['rules']) 
                nl=item[0].metadata['question']
                q=item[0].metadata['query']
                score=item[1]
                scores.append(score)
                nl2sql_pairs.append(
                    {'question':nl, 'query':q,'score':score}
                )
            # print('###########') 
            
            examples_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore,
                k = max_shot,
            )
            examples_prompt = PromptTemplate(
                input_variables=["question","query"],
                template="### Question\n{question}\n### SQL query\n{query}",
            )  
            prompt_template = few_shot_template_examples(examples_prompt, examples_selector)
            llm_chain= LLMChain(llm=llm, prompt= prompt_template)
            
            sql = None
            
            
            llm_inputs = {
                "input": question,
                "table_info": ddl,
                "rules":[r["extracted_rules"] for r in rules_db_ids if r['db_id']==db_id][-1]
                # "rules":rules
            }

            prompts = llm_chain.prep_prompts([llm_inputs])
           
            prompt_strings = [p for p in prompts[0]][0].to_string() 
            print(prompt_strings)
            # check the length:
            # Write function to take string input and return number of tokens
            num_tokens = num_tokens_from_string(prompt_strings, args.model_name)

            print(f"Starting  generation item {iddx} #input-tokens {num_tokens}")
            tic =time.perf_counter()
            while sql is None:
                try:
                    sql = llm_chain.run(**llm_inputs)
                    print(sql)
                except Exception as e:
                    print(str(e))
                    time.sleep(3)
                    pass
            toc = time.perf_counter()
            
            process_time=toc-tic
            print(f"Process Time= {process_time:0.4f} second")
            print('##############################')                    
            r = {
                "i": iddx,
                "question": question,
                "query": query,
                "db_id" : db_id,
                "generated_query":sql,
                "meta_data":nl2sql_pairs,
                "prompt": str(prompt_strings),
                "process_time":process_time,
                "num_tokens":num_tokens,
                "hardness":hardness,
                "scores": scores,
                "lang":lang
                }

            all_results.append(r)
            

            # Sets file's current position at offset.
            output.seek(0)
            json.dump(all_results, output, indent=2)

    with open(results_file) as f:
        all_results = json.load(f)
    
    with open(f'{args.data_path}/outputs/{args.model_name}/{max_shot}-shot-examples-multi-train-70-30-test_old_en_emb_run#{run}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['db_id',"question","query","generated_query","hardness","lang","process_time","Max_Score","Avg_Score"])
        for item in all_results:
            Max_Score= min(item['scores'])
            Avg_Score= sum(item['scores'])/len(item['scores'])
            writer.writerow([item['db_id'],item["question"],item["query"].strip(), item['generated_query'].strip(),item['hardness'].strip(),item['lang'].strip(),item['process_time'],Max_Score,Avg_Score])

    return file.name
