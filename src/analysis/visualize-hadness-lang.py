import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

data = {
# zero-shot mixtral
    "EN-zero-shot-mixtral": {
    "unknown": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "18.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "100.00",
        "std": "0.00"
      },
      
      "examples": {
        "mean": "2.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "41.67",
        "std": "5.89"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "4.17",
        "std": "0.00"
      },
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
  "DE-zero-shot-mixtral": {
    "unknown": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },

      "examples": {
        "mean": "41.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "0.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "12.50",
        "std": "0.00"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
# few shot Mixtral
  "EN-few-shot-mixtral": {
    "unknown": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "18.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "50.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "2.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "45.00",
        "std": "6.12"
      },

      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "25.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
  "DE-few-shot-mixtral": {
    "unknown": {
      "hard": {
        "mean": "24.39",
        "std": "0.00"
      },

      "examples": {
        "mean": "41.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "0.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "45.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "25.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
# zero-shot gpt
    "EN-zero-shot-gpt": {
    "unknown": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "18.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "100.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "2.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "29.17",
        "std": "5.89"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
  "DE-zero-shot-gpt": {
    "unknown": {
      "hard": {
        "mean": "4.88",
        "std": "0.00"
      },
      "examples": {
        "mean": "41.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "0.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "37.50",
        "std": "0.00"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "22.22",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "29.17",
        "std": "0.00"
      },
     
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
# few-shot gpt
  "EN-few-shot-gpt": {
    "unknown": {
      "hard": {
        "mean": "27.78",
        "std": "0.00"
      },
      "examples": {
        "mean": "18.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "100.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "2.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "75.00",
        "std": "0.00"
      },
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "11.11",
        "std": "0.00"
      },
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "29.17",
        "std": "2.50"
      },
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  },
  "DE-few-shot-gpt": {
    "unknown": {
      "hard": {
        "mean": "44.88",
        "std": "1.20"
      },
     
      "examples": {
        "mean": "41.00",
        "std": "0.00"
      }
    },
    "easy": {
      "hard": {
        "mean": "0.00",
        "std": "0.00"
      },
     
      "examples": {
        "mean": "0.00",
        "std": "0.00"
      }
    },
    "medium": {
      "hard": {
        "mean": "50.00",
        "std": "0.00"
      },
      
      "examples": {
        "mean": "8.00",
        "std": "0.00"
      }
    },
    "hard": {
      "hard": {
        "mean": "44.44",
        "std": "0.00"
      },
    
      "examples": {
        "mean": "9.00",
        "std": "0.00"
      }
    },
    "extra": {
      "hard": {
        "mean": "50.00",
        "std": "0.00"
      },
      
      "examples": {
        "mean": "24.00",
        "std": "0.00"
      }
    }
  }
 
}

hardness=['easy','medium','hard','extra','unknown']
labels = list(data.keys())
easy_means = [float(data[label]['easy']['hard']['mean']) for label in labels]
medium_means = [float(data[label]['medium']['hard']['mean']) for label in labels]
hard_means = [float(data[label]['hard']['hard']['mean']) for label in labels]
extra_means = [float(data[label]['extra']['hard']['mean']) for label in labels]
unknown_means = [float(data[label]['unknown']['hard']['mean']) for label in labels]



easy_std = [float(data[label]['easy']['hard']['std']) for label in labels]
medium_std = [float(data[label]['medium']['hard']['std']) for label in labels]
hard_std = [float(data[label]['hard']['hard']['std']) for label in labels]
extra_std = [float(data[label]['extra']['hard']['std']) for label in labels]
unknown_std = [float(data[label]['unknown']['hard']['std']) for label in labels]


print(labels)
print(easy_means)
print(medium_means)
print(hard_means)
print(extra_means)
print(unknown_means)



print(easy_std)
print(medium_std)
print(hard_std)
print(extra_std)
print(unknown_std)

# use this y coordinate to add labels
means = [
    item
    for grp in zip(easy_means, medium_means, hard_means, extra_means, unknown_means)
    for item in grp
]

easy_examples = [int(float(data[label]["easy"]["examples"]["mean"])) for label in labels]
medium_examples = [int(float(data[label]["medium"]["examples"]["mean"])) for label in labels]
hard_examples = [int(float(data[label]["hard"]["examples"]["mean"])) for label in labels]
extra_examples = [int(float(data[label]["extra"]["examples"]["mean"])) for label in labels]
unknown_examples = [
    int(float(data[label]["unknown"]["examples"]["mean"])) for label in labels
]
# lets say this is the text labels
examples = [
    item
    for grp in zip(
        easy_examples, medium_examples, hard_examples, extra_examples, unknown_examples
    )
    for item in grp
]


x = np.arange(len(labels)) # the label locations
width = 0.175  # adjusted width of the bars

fig, ax = plt.subplots()

def addlabels(y, text, fontsize=6):
    for i in range(len(y)):
        plt.text(i * 0.2 - 0.4, y[i] // 2, text[i], ha="center", fontsize=fontsize)

# rects1 = ax.bar(x - width*2, easy_means, width, yerr=easy_std, color='mediumaquamarine', label='Easy', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
# rects2 = ax.bar(x - width, medium_means, width, yerr=medium_std, color='bisque', label='Medium', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
# rects3 = ax.bar(x, hard_means, width, yerr=hard_std, color='lightsalmon', label='Hard', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
# rects4 = ax.bar(x + width, extra_means, width, yerr=extra_std, color='red', label='Extra hard', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
# rects5 = ax.bar(x + width*2, unknown_means, width, yerr=unknown_std, color='lightsteelblue', label='Unknown', ecolor='black', capsize=3,edgecolor = "black")  # specify color here


rects1 = ax.bar(x - width*2, easy_means, width, color='mediumaquamarine', label='Easy', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
rects2 = ax.bar(x - width, medium_means, width,  color='bisque', label='Medium', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
rects3 = ax.bar(x, hard_means, width,  color='lightsalmon', label='Hard', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
rects4 = ax.bar(x + width, extra_means, width,  color='red', label='Extra hard', ecolor='black', capsize=3,edgecolor = "black")  # specify color here
rects5 = ax.bar(x + width*2, unknown_means, width, color='lightsteelblue', label='Unknown', ecolor='black', capsize=3,edgecolor = "black")  # specify color here


ax.set_ylabel('Execution Accuracy (Strict EA)')
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
ax.tick_params("x", length=0)
ax.set_xlim(-.75, 7.75)


# Add ticks and labels for the shelves.
shelf_ax = ax.secondary_xaxis(location=0)
shelf_ax.set_xticks([i *2+ 0.5 for i in range(4)], labels=["Zero-shot", "Few-shot"] * 2)
shelf_ax.tick_params("x", length=20)

# Add ticks and labels for the rooms.
room_ax = ax.secondary_xaxis(location=0)
room_ax.set_xticks([1.5, 5.5], labels=["Mixtral","GPT-3.5"])
room_ax.tick_params("x", length=30)


# Long ticks with no labels to separate the rooms.
room_sep_ax = ax.secondary_xaxis(location=0)
room_sep_ax.set_xticks([-.75, 3.5, 7.75], ["", "", ""])
room_sep_ax.tick_params("x", length=45)

# addlabels(means, examples)

ax.legend(loc='upper right')
ax.grid(color='gray', linestyle='dashed',axis='y')

ax.set_ylim([0, 110])  # Set the start of Y-axis to 30
plt.yticks(np.arange(0, 110, 10))  # change y-ticks frequency here

fig.tight_layout()
plt.savefig('means_by_lang-hardness.png')  # Save the figure as PNG
plt.show()