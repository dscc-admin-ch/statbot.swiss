import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

data = {
# zero-shot mixtral
   "EN ": {
    "hard": {
      "mean": "10.39",
      "std": "0.77"
    },
    "soft": {
      "mean": "10.39",
      "std": "0.77"
    },
    "partial": {
      "mean": "12.02",
      "std": "0.77"
    },
    "number_of_examples": {
      "mean": "61.00"
    }
  },
  "DE ": {
    "hard": {
      "mean": "2.44",
      "std": "0.00"
    },
    "soft": {
      "mean": "2.44",
      "std": "0.00"
    },
    "partial": {
      "mean": "2.44",
      "std": "0.00"
    },
    "number_of_examples": {
      "mean": "82.00"
    }
  },
# few shot Mixtral
  "EN_": {
    "hard": {
      "mean": "22.29",
      "std": "0.80"
    },
    "soft": {
      "mean": "27.21",
      "std": "0.80"
    },
    "partial": {
      "mean": "30.49",
      "std": "0.80"
    },
    "number_of_examples": {
      "mean": "61.00"
    }
  },
  "DE_": {
    "hard": {
      "mean": "32.93",
      "std": "0.00"
    },
    "soft": {
      "mean": "41.46",
      "std": "0.00"
    },
    "partial": {
      "mean": "43.90",
      "std": "0.00"
    },
    "number_of_examples": {
      "mean": "82.00"
    }
  },
# zero-shot gpt
"EN#": {
    "hard": {
      "mean": "8.75",
      "std": "0.77"
    },
    "soft": {
      "mean": "8.75",
      "std": "0.77"
    },
    "partial": {
      "mean": "10.93",
      "std": "0.77"
    },
    "number_of_examples": {
      "mean": "61.00"
    }
  },
  "DE#": {
    "hard": {
      "mean": "17.07",
      "std": "0.00"
    },
    "soft": {
      "mean": "17.48",
      "std": "0.58"
    },
    "partial": {
      "mean": "23.17",
      "std": "0.00"
    },
    "number_of_examples": {
      "mean": "82.00"
    }
  },
# few-shot gpt
  "EN": {
    "hard": {
      "mean": "34.98",
      "std": "0.77"
    },
    "soft": {
      "mean": "39.34",
      "std": "0.00"
    },
    "partial": {
      "mean": "39.89",
      "std": "0.77"
    },
  },
  "DE": {
    "hard": {
      "mean": "47.15",
      "std": "0.58"
    },
    "soft": {
      "mean": "54.47",
      "std": "0.58"
    },
    "partial": {
      "mean": "57.32",
      "std": "0.00"
    },
  }
}

# "number_of_examples": {
#       "mean": "61.00"
#     }
# "number_of_examples": {
#       "mean": "82.00"
#     }

# "En": {
  #   "hard": {
  #     "mean": "18.03",
  #     "std": "1.04"
  #   },
  #   "soft": {
  #     "mean": "24.59",
  #     "std": "1.04"
  #   },
  #   "partial": {
  #     "mean": "30.17",
  #     "std": "1.67"
  #   },
  #   "number_of_examples": {
  #     "mean": "61.00"
  #   }
  # },
  # "De": {
  #   "hard": {
  #     "mean": "24.39",
  #     "std": "0.00"
  #   },
  #   "soft": {
  #     "mean": "35.37",
  #     "std": "0.00"
  #   },
  #   "partial": {
  #     "mean": "40.24",
  #     "std": "0.00"
  #   },
  #   "number_of_examples": {
  #     "mean": "82.00"
  #   }
  # },

labels = list(data.keys())
hards_means = [float(data[label]['hard']['mean']) for label in labels]
soft_means = [float(data[label]['soft']['mean']) for label in labels]
partial_means = [float(data[label]['partial']['mean']) for label in labels]
hards_std = [float(data[label]['hard']['std']) for label in labels]
soft_std = [float(data[label]['soft']['std']) for label in labels]
partial_std = [float(data[label]['partial']['std']) for label in labels]





x = np.arange(len(labels)) # the label locations
width = .8 # adjusted width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x , hards_means, width, yerr=hards_std, color='darkkhaki', ecolor='black', capsize=3,edgecolor = "black")  # specify color here

#rects1 = ax.bar(x - width, hards_means, width, yerr=hards_std, color='salmon', label='Hard EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#rects2 = ax.bar(x, soft_means, width, yerr=soft_std, color='papayawhip', label='Soft EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#rects3 = ax.bar(x + width, partial_means, width, yerr=partial_std, color='mediumaquamarine', label='Partial EA', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
#lightsteelblue

ax.set_ylabel('Execution Accuracy (Hard EA)')
# ax.set_title('Language')


ax.set_xticks(x)
ax.set_xticklabels(['EN','DE','EN','DE','EN','DE','EN','DE'])


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

ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.tick_params("x", length=2)
ax.set_xlim(-0.5, 7.5)


# Add ticks and labels for the shelves.
shelf_ax = ax.secondary_xaxis(location=0)
shelf_ax.set_xticks([i *2+ 0.5 for i in range(4)], labels=["Zero-shot", "Few-shot"] * 2)
shelf_ax.tick_params("x", length=15)

# Add ticks and labels for the rooms.
room_ax = ax.secondary_xaxis(location=0)
room_ax.set_xticks([1.5, 5.5], labels=["Mixtral","GPT-3.5-Turbo"])
room_ax.tick_params("x", length=25)

# Long ticks with no labels to separate the rooms.
room_sep_ax = ax.secondary_xaxis(location=0)
room_sep_ax.set_xticks([-0.5, 3.5, 7.5], ["", "", ""])
room_sep_ax.tick_params("x", length=40)

#ax.legend()
ax.grid(color='gray', linestyle='dashed',axis='y')

ax.set_ylim([0, 60])  # Set the start of Y-axis to 30
plt.yticks(np.arange(0, 60, 5))  # change y-ticks frequency here

fig.tight_layout()
plt.savefig('means_by_lang.png')  # Save the figure as PNG
plt.show()



