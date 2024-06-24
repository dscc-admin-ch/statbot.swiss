import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

def addlabels(y, text, fontsize=6):
    for i in range(len(y)):
        plt.text(i * 0.2 - 0.4, y[i] // 2, text[i], ha="center", fontsize=fontsize)


data = {
# zero-shot mixtral
   "zero-shot-mixtral-en": {
    "hard": {
      "mean": "10.39",
      "std": "0.77"
    },
  },
  "zero-shot-mixtral-de ": {
    "hard": {
      "mean": "2.44",
      "std": "0.00"
    }
  },
# few shot Mixtral
  "few-shot-mixtral-en": {
    "hard": {
      "mean": "22.29",
      "std": "0.80"
    },
  },
  "few-shot-mixtral-de": {
    "hard": {
      "mean": "32.93",
      "std": "0.00"
    },
  },
# zero-shot gpt
"zero-shot-gpt-en": {
    "hard": {
      "mean": "8.75",
      "std": "0.77"
    },
  },
  "zero-shot-gpt-de": {
    "hard": {
      "mean": "17.07",
      "std": "0.00"
    }
  },
# few-shot gpt
  "few-shot-gpt-en": {
    "hard": {
      "mean": "34.43",
      "std": "0.00"
    }
  },
  "few-shot-gpt-de": {
    "hard": {
      "mean": "46.83",
      "std": "0.58"
    },
}

}


labels = list(data.keys())
print(labels)
hards_means = [float(data[label]['hard']['mean']) for label in labels]

hards_std = [float(data[label]['hard']['std']) for label in labels]

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center')
 


x = np.arange(len(labels)/2) # the label locations
width = .3 # adjusted width of the bars

fig, ax = plt.subplots()

print(hards_means)


## to visualize the result in for each zero-shot and few-shot based on EN and DE for Mixtral and GPT

rects1 = ax.bar(x - width, hards_means[0::2], width, color='salmon', label='EN', ecolor='black', capsize=3,edgecolor = "black") 
rects2 = ax.bar(x , hards_means[1::2], width, color='lightsteelblue', label='DE',ecolor='black', capsize=3,edgecolor = "black")  # specify color here
 
 
 # specify color here
#rects2 = ax.bar(x, soft_means, width, yerr=soft_std, color='papayawhip', label='Soft EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#rects3 = ax.bar(x + width, partial_means, width, yerr=partial_std, color='mediumaquamarine', label='Partial EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#lightsteelblue

ax.set_ylabel('Execution Accuracy (Strict EA)')
# ax.set_title('Language')


ax.set_xticks([-.15,0.85,1.85,2.85])
ax.set_xticklabels(['Zero-shot','Few-shot','Zero-shot','Few-shot'])

# use this y coordinate to add labels


# ax.set_xticklabels(['Method A', 'Method B'])
# ax.set_xticks(label_x, minor=True)
# ax.set_xticklabels(label_mode, minor=True)
# ax.tick_params(axis='x', which='minor', length=0)


# ax.tick_params(axis='x', which='major', length=10, width=1)

ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
ax.tick_params("x", length=5)
ax.set_xlim(-.5, 3.5)


# Add ticks and labels for the shelves.
shelf_ax = ax.secondary_xaxis(location=0)
xticks=[.33,2.33]
print(xticks)
shelf_ax.set_xticks(xticks, labels=["Mixtral", "GPT-3.5-Turbo"])
shelf_ax.tick_params("x", length=15)

# # Add ticks and labels for the rooms.
# room_ax = ax.secondary_xaxis(location=0)
# room_ax.set_xticks([1.75, 3.5], labels=["",""])
# room_ax.tick_params("x", length=25)

# Long ticks with no labels to separate the rooms.
room_sep_ax = ax.secondary_xaxis(location=0)
room_sep_ax.set_xticks([-0.5, 1.33, 3.5], ["", "", ""])
room_sep_ax.tick_params("x", length=40)

ax.legend()
ax.grid(color='gray', linestyle='dashed',axis='y')

ax.set_ylim([0, 50])  # Set the start of Y-axis to 30
plt.yticks(np.arange(0, 50, 5))  # change y-ticks frequency here

fig.tight_layout()
plt.savefig('means_by_lang-HEA.png')  # Save the figure as PNG
plt.show()



