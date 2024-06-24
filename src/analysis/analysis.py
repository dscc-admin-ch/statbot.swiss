import json
import os
import numpy as np

#path='dataset/statbot/llm_results/Mixtral/multilingual-emb'
#path='data/statbot/gpt-3.5-turbo-16k/runs/zero-shots'
path='data/statbot/mixtral/mulitlingual-embeddings'
json_files = [f for f in os.listdir(path) if not f.endswith('.json')]
#json_files = [f for f in os.listdir(path) if not f.endswith('.json')]

results={k:{}for k in [0]}
for json_file in json_files:
    print(json_file)
    with open(os.path.join(path, json_file))as f:
        d = json.load(f)
        hards=[]
        softs=[]
        partials=[]
        shot= json_file.split("-")[0]
        for r in d:
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
    print(results)
    with open(f'{path}/random.json',"w") as llm:
                json.dump(results, llm, indent=2)
           
        