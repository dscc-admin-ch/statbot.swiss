import csv
import os

target="data/statbot/mixtral/mulitlingual-embeddings/csv-outputs"
for i in [1,2,3,4,5,6,8]:
    for j in [0,1,2,3,4]:
        with open(f'{target}/{i}-shot_Mixtral-8x7B-Instruct-v0.1-train-test-old-run#{j}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
       