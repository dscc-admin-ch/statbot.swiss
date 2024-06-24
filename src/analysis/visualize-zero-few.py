import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


data={
  "zero-shot-mixtral": {
    "hard": {
      "mean": "5.82",
      "std": "0.33"
    },
    "soft": {
      "mean": "5.82",
      "std": "0.33"
    },
    "partial": {
      "mean": "6.52",
      "std": "0.33"
    }
  },
  "few-shot-mixtral":{
    "hard": {
      "mean": "28.39",
      "std": "0.34"
    },
    "soft": {
      "mean": "35.38",
      "std": "0.34"
    },
    "partial": {
      "mean": "38.18",
      "std": "0.34"
    }
     },
    "zero-shot-gpt":{
    "hard": {
      "mean": "13.52",
      "std": "0.33"
    },
    "soft": {
      "mean": "13.76",
      "std": "0.33"
    },
    "partial": {
      "mean": "17.95",
      "std": "0.33"
    }
    },
    "few-shot-gpt-5": {
    "hard": {
      "mean": "41.68",
      "std": "0.56"
    },
    "soft": {
      "mean": "48.25",
      "std": "0.44"
    },
    "partial": {
      "mean": "50.07",
      "std": "0.34"
    }
  }
}

labels = list(data.keys())
print(labels)
hards_means = [float(data[label]['hard']['mean']) for label in labels]
soft_means = [float(data[label]['soft']['mean']) for label in labels]
partial_means = [float(data[label]['partial']['mean']) for label in labels]
hards_std = [float(data[label]['hard']['std']) for label in labels]
soft_std = [float(data[label]['soft']['std']) for label in labels]
partial_std = [float(data[label]['partial']['std']) for label in labels]




x = np.arange(len(labels)/2) # the label locations
width = .4 # adjusted width of the bars

fig, ax = plt.subplots()

print(hards_means)

print(hards_means)
## to visualize the result in for each zero-shot and few-shot based on EN and DE for Mixtral and GPT
#yerr=hards_std[0::2]

rects1 = ax.bar(x - width, hards_means[0::2], width,yerr=hards_std[0::2], color='lightsteelblue', label='Zero-shot', ecolor='black', capsize=3, edgecolor = "black") 
# yerr=hards_std[1::2]
rects2 = ax.bar(x , hards_means[1::2], width, yerr=hards_std[1::2], color='papayawhip', label='Few-shot', ecolor='black', capsize=3, edgecolor = "black") 
#rects2 = ax.bar(x , hards_means[1::2], width, yerr=hards_std[1::2], color='lightsteelblue', label='DE',ecolor='black', capsize=3,edgecolor = "black")  # specify color here
 
 
 # specify color here
#rects2 = ax.bar(x, soft_means, width, yerr=soft_std, color='papayawhip', label='Soft EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#rects3 = ax.bar(x + width, partial_means, width, yerr=partial_std, color='mediumaquamarine', label='Partial EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#lightsteelblue

ax.set_ylabel('Execution Accuracy (Strict EA)')
# ax.set_title('Language')


ax.set_xticks([-.2,.8])
ax.set_xticklabels(['Mixtral','GPT-3.5'])


# label_x, label_mode = [], []

# for rect, idx in zip(ax.patches, ['En','De','En ','De ']):
#     print(rect, idx)
#     label_mode.append(idx)
#     label_x.append(rect.xy[0] + (rect.get_width()))

# ax.set_xticklabels(['Method A', 'Method B'])
# ax.set_xticks(label_x, minor=True)
# ax.set_xticklabels(label_mode, minor=True)
# ax.tick_params(axis='x', which='minor', length=0)


# ax.tick_params(axis='x', which='major', length=10, width=1)

# ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
# ax.tick_params("x", length=5)
# ax.set_xlim(-.5, 3.5)


# # Add ticks and labels for the shelves.
# shelf_ax = ax.secondary_xaxis(location=0)
# xticks=[.33,2.33]
# print(xticks)
# shelf_ax.set_xticks(xticks, labels=["Mixtral", "GPT-3.5-Turbo"])
# shelf_ax.tick_params("x", length=15)

# # Add ticks and labels for the rooms.
# room_ax = ax.secondary_xaxis(location=0)
# room_ax.set_xticks([1.75, 3.5], labels=["",""])
# room_ax.tick_params("x", length=25)

# Long ticks with no labels to separate the rooms.
room_sep_ax = ax.secondary_xaxis(location=0)
room_sep_ax.set_xticks([-.69, 0.3,1.29], ["", "", ""])
room_sep_ax.tick_params("x", length=20)

ax.legend(loc="upper left")
ax.grid(color='gray', linestyle='dashed',axis='y')

ax.set_ylim([5, 45])  # Set the start of Y-axis to 30
plt.yticks(np.arange(5, 45, 5))  # change y-ticks frequency here

fig.tight_layout()
plt.savefig('means_by_zero-few.png')  # Save the figure as PNG
plt.show()



