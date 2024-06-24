
import json
import numpy as np
import matplotlib.pyplot as plt
few_shots_5_results="data/statbot/gpt-3.5-turbo-16k/runs/multilingual-embeddings/json-outputs/5-shot_with_examples-multil_train_70-30_test_old_run#2.json"

with open(few_shots_5_results,"r") as f:
   data=json.load(f)

dbs={}
for item in data:
   db=item['db_id']
   scores=item['scores']
   if db in dbs:
      dbs[db].extend([np.mean(scores)])
   else:
      dbs[db]=scores   

scores=[]
errors=[]

dbs_scores = {}
dbs_scores_={}
for db in dbs:
   scores.append(np.mean(dbs[db]))
   errors.append(np.std(dbs[db]))
   dbs_scores[db]={'mean':np.mean(dbs[db]), 'std':np.std(dbs[db])}
   dbs_scores_[db]=(np.mean(dbs[db])-np.std(dbs[db]),np.mean(dbs[db])+np.std(dbs[db]))

sorted_dbs=sorted(dbs_scores.items(), key=lambda x:x[1]['mean'])
sorted_dbs_=sorted(dbs_scores_.items(), key=lambda x:x[1][0])
for k,v in sorted_dbs:
   print(k,v)

# for k,v in sorted_dbs_:
#     print(k,v)

x_pos=np.arange(len(dbs))
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, scores, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Similarity scores')
ax.set_xticks(x_pos)
ax.set_xticklabels(dbs.keys())
ax.set_title('')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('db_similarity_scores_in_examples.png')
plt.show()

