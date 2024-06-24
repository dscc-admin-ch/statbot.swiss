import json
# read the dev_result
# with open("dev_results.json") as f:
#     all_results = json.load(f)
#
# # text-davinci-003 0:1034
# text_davinci = all_results[:1034]
# gpt_turbo = all_results[1034:]
#
# with open("text_davinci.json","w") as text_davinci_output,open("gpt_turbo.json","w") as gpt_turbo_output :
#     json.dump(text_davinci, text_davinci_output, indent=2)
#     json.dump(gpt_turbo, gpt_turbo_output, indent=2)
#
# with open("err.json") as f:
#     err = json.load(f)
#
# ids=[]
# for ex in err:
#     ids.append(ex["i"])
#
#
# with open("gpt_turbo_demystified.json") as f:
#     gpt_turbo_results = json.load(f)
#
# gpt_turbo_demystified=[]
# seen=ids
# for ex in gpt_turbo_results:
#     if ex["i"] in seen:
#         continue
#     else:
#         seen.append(ex["i"])
#         gpt_turbo_demystified.append(ex)
#
# with open("gpt_turbo_demystified.json","w") as gpt_turbo_errors:
#     json.dump(gpt_turbo_demystified, gpt_turbo_errors, indent=2)
#
with open("err.json") as f:
    err = json.load(f)
#
# print(len(err), len(gpt_turbo_demystified))

error_types={}
#
#
print("N:",len(err))
examples=[]
numebr_of_errors=0
#
for ex in err:
    if 'error' in ex:
        numebr_of_errors+=1
        err_type = ex['error'].split("\n[SQL")[0].split(":")[0]
        if err_type in error_types:
            error_types[err_type]+= 1
        else:
            error_types[err_type] = 1
        ex['error-type']=err_type
        examples.append(ex)


for k,v in error_types.items():
    print(k,v)

print("numebr_of_errors:", numebr_of_errors)

# for ex in gpt_turbo_results:
#     if 'error' in ex:
#         continue
#     else:
#         err.append(ex)
#
# with open("gpt_turbo_errors.json","w") as gpt_turbo_errors:
#     json.dump(examples, gpt_turbo_errors, indent=2)
