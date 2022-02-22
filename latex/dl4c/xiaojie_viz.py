import os
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import json
import pandas as pd

############## Parameters for graphs by hist_visual ################
# Define the colors and the corresponding value intervals for each color
# e.g. value < 0.2 -> color = lightsteelblue
#      0.2 <= value < 0.4 -> color = cornflowerblue
#      ...
# Make sure that len(COLORS) = len(VALUE_INTERVALS) + 1
# For more colors, please check https://matplotlib.org/stable/gallery/color/named_colors.html

COLORS_BEFORE = ['mistyrose', 'lightcoral', 'red', 'darkred']
COLORS_AFTER = ['lightsteelblue', 'royalblue', 'blue', 'navy']
VALUE_INTERVALS = [0.25, 0.5, 0.75]
BINWIDTH = 0.1
BINS = np.arange(0.0, 1+BINWIDTH, BINWIDTH)


######################################################

def hist_visual(data: dict, model_name, sct, img_dir):
    # data is in the form of data = {'before': [], 'after':[]}

    # compute mean and variance
    data_after = np.array(data['after'])
    mean = np.mean(data_after)
    var = np.var(data_after)

    # decide the graph color based on mean and var values, i.e. deeper colors for mean with larger values
    color_ind = 0
    for i in range(len(VALUE_INTERVALS)):
        if VALUE_INTERVALS[i] <= mean:
            color_ind = i + 1
        else:
            break

    plt.figure(figsize=(10, 9))
    seaborn.set_theme(style="whitegrid")
    plt.hist([data['before'], data['after']], color=[COLORS_BEFORE[color_ind], COLORS_AFTER[color_ind]],
             label=['before', 'after'], alpha=0.9, bins=BINS)
    plt.xticks(BINS)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{model_name} x {sct}", fontsize=20)

    # Save image
    filename = model_name.replace('/', '_')
    plt.savefig(os.path.join(img_dir, '{}_{}.png'.format(filename, sct)))
    print("Image saved for {} x {}".format(model_name, sct))


def box_visual(data: pd.DataFrame, task_name, img_dir):


    plt.figure(figsize=(15, 10))
    seaborn.set_theme(style="whitegrid")
    sns_plot = seaborn.boxplot(data=data, x='sct', y='score', hue='model',  showfliers = False) \
        .set_title(task_name, fontsize=30)
    plt.legend(bbox_to_anchor=(0.7, 0.24), loc='upper left', borderaxespad=0)
    plt.xlabel("Source Code Transformation", fontsize = 14)
    plt.ylabel("Mean Reciprocal Rank", fontsize = 14)
    # plt.show()

    # save graphs
    fig = sns_plot.get_figure()


    fig.savefig(os.path.join(img_dir, '{}.png'.format(task_name)))

    print("Image saved for {} task".format(task_name))


if __name__ == '__main__':
    img_dir = './figs/'
    data_file = './variable_misuse.json'

    models = [
        'microsoft/codebert-base-mlm',
        'microsoft/graphcodebert-base',
        'dbernsohn/roberta-java',
        'huggingface/CodeBERTa-small-v1'
    ]
    scts = [
        'renameTokens',
        'permuteArgumentOrder',
        'swapMultilineNoDeps',
        'addExtraLogging'
    ]

    # Read data
    data = None
    with open(data_file) as json_file:
        data = json.load(json_file)

    ############### Histgram visualization ###################################
    # for m in models:
    #     for sct in scts:
    #         hist_visual(data[m][sct], m, sct, img_dir)

    ############### Box Visualization ########################################
    task_name = 'Variable Misuse'

    df = {'model': [], 'sct': [], 'score': []}
    for model in models:
        print(f'{model} \stackanchor{{Before}}{{After}}  ', end='')
        for sct in scts:
            before = data[model][sct]['before']
            bavg = '{:.4f}'.format(np.mean(before))
            bstd = '{:.4f}'.format(np.std(before))
            after = data[model][sct]['after']
            aavg = '{:.4f}'.format(np.mean(after))
            astd = '{:.4f}'.format(np.std(after))
            print(f' & \stackanchor{{({bavg}, {bstd})}}{{({aavg}, {astd})}}', end='')
            bl = len(data[model][sct]['before'])
            df['model'].extend([model + '(before)'] * bl)
            df['sct'].extend([sct] * bl)
            df['score'].extend(data[model][sct]['before'])
            al = len(data[model][sct]['after'])
            df['model'].extend([model + '(after)'] * al)
            df['sct'].extend([sct] * al)
            df['score'].extend(data[model][sct]['after'])
        print('\\\\\\\\')
        print('')


    df = pd.DataFrame(data=df)
    box_visual(df, task_name, img_dir)

def mean_std_dev(model, l1, l2):
    print(f'{np.mean(l1)}, {np.std(l2)}')

    # microsoft/codebert-base-mlm \stackanchor{Before}{After}  & \stackanchor{(1.366, 998)}{(1.366, 998)}                   & \stackanchor{(1.366, 998)}{(1.366, 998)}                   & \stackanchor{(1.366, 998)}{(1.366, 998)}                   & \stackanchor{(1.366, 998)}{(1.366, 998)}                   \\\\
