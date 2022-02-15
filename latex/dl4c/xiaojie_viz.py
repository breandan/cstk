import json
import os
import numpy as np
import seaborn
import matplotlib.pyplot as plt

############### Parameters for graphs ################
# Define the colors and the corresponding value intervals for each color
# e.g. value < 0.2 -> color = lightsteelblue
#      0.2 <= value < 0.4 -> color = cornflowerblue
#      ...
# Make sure that len(COLORS) = len(VALUE_INTERVALS) + 1
# For more colors, please check https://matplotlib.org/stable/gallery/color/named_colors.html

COLORS = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'blue']
VALUE_INTERVALS = [0.2, 0.4, 0.6]

BINWIDTH = 0.02
######################################################


def visualization(data, model_name, sct, img_dir):

    # compute mean and variance
    data = np.array(data)
    mean = np.mean(data)
    var = np.var(data)

    # decide the graph color based on mean and var values, i.e. deeper colors for mean with larger values
    color = COLORS[0]
    for i in range(len(VALUE_INTERVALS)):
        if VALUE_INTERVALS[i] <= mean:
            color = COLORS[i + 1]
        else:
            break

    plt.figure()
    sns_plot = seaborn.histplot(data=data, color=color, binwidth=BINWIDTH)\
        .set_title('{} x {}\n mean:{:.3f}, var:{:.3f}'.format(model_name, sct, mean, var))
    # plt.show()

    # save graphs
    fig = sns_plot.get_figure()
    filename = model_name.replace('/', '_')
    fig.savefig(os.path.join(img_dir, '{}_{}.png'.format(filename, sct)))
    print("Image saved for {} x {}".format(model_name, sct))


img_dir = './'
data_file = 'data.json'
models = ['microsoft/codebert-base-mlm', 'microsoft/graphcodebert-base', 'dbernsohn/roberta-java']
scts = ['renameTokens', 'permuteArgumentOrder', 'swapMultilineNoDeps', 'addExtraLogging']

data = None
with open(data_file) as json_file:
    data = json.load(json_file)

for m in models:
    for sct in scts:
        visualization(data[m][sct]['after'], m, sct, img_dir)
