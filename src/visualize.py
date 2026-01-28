import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import matplotlib.patches as mpatches
import itertools
from upsetplot import UpSet, generate_counts
from copy import deepcopy
from d3blocks import D3Blocks
from pdftopng import pdftopng
import holoviews as hv
hv.extension("bokeh")

def clean_key(key: str) -> str:
        # Duplicate keys to handle separately:
        # Key philosophy maps to ['philosophy_test', 'philosophy-test-MMLUPRO']
        # Key psychology maps to ['high_school_psychology_test', 'psychology-test-MMLUPRO']
        # Key bio maps to ['college_biology_test', 'high_school_biology_test', 'biology-test-MMLUPRO']
        # Key computer sci maps to ['high_school_computer_science_test', 'computer science-test-MMLUPRO', 'college_computer_science_test']
        # Key chemistry maps to ['high_school_chemistry_test', 'college_chemistry_test', 'chemistry-test-MMLUPRO']
        # Key physics maps to ['high_school_physics_test', 'college_physics_test', 'physics-test-MMLUPRO']
        # Key math maps to ['college_mathematics_test', 'math-test-MMLUPRO', 'high_school_mathematics_test']
        
        if key == 'philosophy_test':
            return 'philosophy (MMLU)'
        elif key == 'philosophy-test-MMLUPRO':
            return 'philosophy (MMLU-PRO)'
        elif key == 'high_school_psychology_test':
            return 'psychology (MMLU)'
        elif key == 'psychology-test-MMLUPRO':
            return 'psychology (MMLU-PRO)'
        elif key == 'college_biology_test':
            return 'college biology (MMLU)'
        elif key == 'high_school_biology_test':
            return 'HS biology (MMLU)'
        elif key == 'biology-test-MMLUPRO':
            return 'biology (MMLU-PRO)'
        elif key == 'high_school_computer_science_test':
            return 'HS comp sci (MMLU)'
        elif key == 'computer science-test-MMLUPRO':
            return 'comp sci (MMLU-PRO)'
        elif key == 'college_computer_science_test':
            return 'college comp sci (MMLU)'
        elif key == 'high_school_chemistry_test':
            return 'HS chemistry (MMLU)'
        elif key == 'college_chemistry_test':
            return 'college chemistry (MMLU)'
        elif key == 'chemistry-test-MMLUPRO':
            return 'chemistry (MMLU-PRO)'
        elif key == 'high_school_physics_test':
            return 'HS physics (MMLU)'
        elif key == 'college_physics_test':
            return 'college physics (MMLU)'
        elif key == 'physics-test-MMLUPRO':
            return 'physics (MMLU-PRO)'
        elif key == 'college_mathematics_test':
            return 'college math (MMLU)'
        elif key == 'math-test-MMLUPRO':
            return 'math (MMLU-PRO)'
        elif key == 'high_school_mathematics_test':
            return 'HS math (MMLU)'
        
        
        banned_words = ['test', 'MMLUPRO', 'high', 'middle', 'primary', 'school', 'college', 'university', 'level', 'difficulty']
        
        for word in banned_words:
            key = key.replace('-'+word, '')
            key = key.replace(word+'-', '')
            key = key.replace(word, '')
        
        key = key.replace('electrical', 'elec').replace('electronic', 'elec').replace('engineering', 'eng').replace('technology', 'tech')
        key = key.replace('mathematics', 'math').replace('science', 'sci')
        key = key.replace('biology', 'bio').replace('history', 'hist').replace('literature', 'lit')
        key = key.replace('language', 'lang').replace('english', 'eng').replace('chinese', 'chi')
        key = key.replace('probability', 'prob').replace('statistics', 'stat').replace('algebra', 'alg')
        key = key.replace('government', 'gov').replace('politics', 'pol').replace('economics', 'econ')
        key = key.replace('medicine', 'med')
        
        key = key.replace('_', ' ').replace('  ', ' ').replace('  ', ' ').strip().lower()
        
        final_mapping = {
            'mc 500': 'story compreh (MCTest)',
            'arc hard': 'general science (ARC Hard)',
            'arc easy-1118': 'general science (ARC Easy)',
            'obqa-1k': 'general science (OBQA)',   
        }
        
        return final_mapping.get(key, key)

def draw_boxplot():

    # Create the subplot structure for 5 keys, each with 2 configurations (Set 1 and Set 2)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.5)

    with open("raw_stats.json", 'r') as f:
        data = json.load(f)

    hkeys = ["values_for_each", "positions_for_each"]
    keys = list(data["values_for_each"].keys())

    for _, hkey in enumerate(hkeys):
        for i, key in enumerate(keys):
            assert len(data[hkey][key]) == 3
            ax = axes[int(key)-1, _]

            # Boxplot
            box = ax.boxplot(data[hkey][key], labels=["8B", "70B", "405B-int4"])
            
            ax.set_title(f'{hkey.split("_")[0].capitalize()} at Level {key}')
            ax.grid(True)

    plt.show()
    plt.savefig("boxplot.pdf")


def draw_errorbars():

    # Create the subplot structure for 5 keys, each with 2 configurations (Set 1 and Set 2)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.5)

    with open("raw_stats.json", 'r') as f:
        data = json.load(f)

    hkeys = ["values_for_each", "positions_for_each"]
    keys = list(data["values_for_each"].keys())

    for _, hkey in enumerate(hkeys):
        for i, key in enumerate(keys):
            assert len(data[hkey][key]) == 3
            ax = axes[int(key)-1, _]

            # Calculating the mean and standard deviation for error bars
            means = [np.mean(values) for values in data[hkey][key]]
            std_devs = [np.std(values) for values in data[hkey][key]]
            std_errs = [std_dev / np.sqrt(len(values)) for values, std_dev in zip(data[hkey][key], std_devs)]
            x_positions = [1, 2, 3]

            # Plotting means with error bars
            ax.errorbar(x_positions, means, yerr=std_errs, fmt='o', color='red', capsize=5)

            ax.set_title(f'{hkey.split("_")[0].capitalize()} at Level {key}')
            ax.grid(True)

    plt.show()
    plt.savefig("errorbars.pdf")
    

def draw_value_curves():
    
    with open("raw_stats.json", 'r') as f:
        data = json.load(f)
    
    hkey = "values_for_each"
    keys = sorted(list(data[hkey].keys()), key=lambda x: np.mean(data[hkey][x]))
    
    means = [[], [], []]
    std_errs = [[], [], []]
    
    for i in range(3):
        for key in keys:
            values = data[hkey][key][i]
            means[i].append(np.mean(values))
            std_errs[i].append(np.std(values) / np.sqrt(len(values)))
    
    means = np.array(means)
    std_errs = np.array(std_errs)
    
    # dashed lines for 8B and 70B
    # plt.plot(keys, means[0], 'o--', label='8B', color='red')
    plt.scatter(keys, means[0], label='8B', color='red')
    plt.errorbar(keys, means[0], yerr=std_errs[0], fmt='o', capsize=5, color='red')
    plt.fill_between(keys, (means[0]-std_errs[0]), (means[0]+std_errs[0]), color='red', alpha=0.1)
    
    # plt.plot(keys, means[1], 'o--', label='70B', color='blue')
    plt.scatter(keys, means[1], label='70B', color='blue')
    plt.errorbar(keys, means[1], yerr=std_errs[1], fmt='o', capsize=5, color='blue')
    plt.fill_between(keys, (means[1]-std_errs[1]), (means[1]+std_errs[1]), color='blue', alpha=0.1)
    
    # plt.plot(keys, means[2], 'o-', label='405B', color='green')
    plt.scatter(keys, means[2], label='405B', color='green')
    plt.errorbar(keys, means[2], yerr=std_errs[2], fmt='o', capsize=5, color='green')
    plt.fill_between(keys, means[2]-std_errs[2], means[2]+std_errs[2], color='green', alpha=0.1)
    
    plt.xlabel('Difficulty Level')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.title('Mean Value Curves ("values_for_each" in Previous Plots) for Different Difficulty Levels', fontsize=10)
    
    # symlog scale for y-axis
    plt.yscale('symlog')
    
    plt.show()
    plt.savefig("value_curves.pdf")


def draw_value_curves_subplots():
    
    N=2
    M=4
    
    # N*M subplots, each with len(keys)/(N*M) keys
    
    with open("raw_stats.json", 'r') as f:
        data = json.load(f)
        
    hkey = "values_for_each"
    keys = sorted(list(data[hkey].keys()), key=lambda x: np.mean(data[hkey][x]))
    print(len(keys))
    
    fig, axes = plt.subplots(N, M, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5)
    
    # reduce top margin
    plt.subplots_adjust(top=0.98)
    plt.subplots_adjust(bottom=0.15)
    
    
    simp_map = defaultdict(list)
    for key in keys:
        simp_map[clean_key(key)].append(key)
    
    for cleaned_key, mapped_keys in simp_map.items():
        if len(mapped_keys) > 1:
            print(f'Key {cleaned_key} maps to {mapped_keys}')
    
    for i in range(N):
        for j in range(M):
            ax = axes[i, j]
            start = (i*M+j) * len(keys) // (N*M)
            end = (i*M+j+1) * len(keys) // (N*M)
            
            cur_keys = keys[start:end]
            cur_keys_cleaned = [clean_key(key) for key in cur_keys]
            
            means = [[], [], []]
            std_errs = [[], [], []]
            
            for key in cur_keys:
                for k in range(3):
                    values = data[hkey][key][k]
                    means[k].append(np.mean(values))
                    std_errs[k].append(np.std(values) / np.sqrt(len(values)))
                    
            means = np.array(means)
            std_errs = np.array(std_errs)
            
            for k in range(3):
                color = ['red', 'blue', 'green'][k]
                ax.scatter([l + (k-1)*0.15 for l in range(start, end)], means[k], label=['8B', '70B', '405B'][k], color=color)
                ax.errorbar([l + (k-1)*0.15 for l in range(start, end)], means[k], yerr=std_errs[k], fmt='o', capsize=5, color=color, label='NA')
                ax.fill_between([l + (k-1)*0.15 for l in range(start, end)], means[k]-std_errs[k], means[k]+std_errs[k], color=color, alpha=0.1, label='NA')
                
            # ax.set_title(f'Subplot {i*M+j+1}')
            ax.set_xticks(range(start, end))
            ax.set_xticklabels(cur_keys_cleaned, rotation=45, ha='right')
            ax.grid(True)
            
    # fig.legend, but only display the first 3 labels
    handles, labels = [], []
    for i in range(N):
        for j in range(M):
            ax = axes[i, j]
            for handle, label in zip(*ax.get_legend_handles_labels()):
                # Skip certain elements (e.g., you can skip tan(x) by checking its label)
                if label in ['8B', '70B', '405B'] and label not in labels:
                    handles.append(handle)
                    labels.append(label)

    axes[0, 0].set_ylabel('Mean Score in Peer Prediction')
    axes[1, 0].set_ylabel('Mean Score in Peer Prediction')
    
    # Add a shared legend outside the subplots; enlarged
    fig.legend(handles[::-1], labels[::-1], loc='center right', ncol=1, fontsize='large')
    
    plt.show()
    plt.savefig("value_curves_subplots.pdf")


def compare_deceptive():
    
    for hkey in ["values_for_each", "positions_for_each"]:
        target = 'Score' if hkey == 'values_for_each' else 'Rank'
        draw_regression = 1
        num_cols = 5
    
        # rows = [
        #     ('./raw_stats-360M-16.json', 1, 'Peer Prediction (n=16)'),
        #     ('./raw_stats-360M-8.json', 1, 'Peer Prediction (n=8)'),
        #     ('./raw_stats-360M-4.json', 1, 'Peer Prediction (n=4)'),
        #     ('./raw_stats-360M-2.json', 1, 'Peer Prediction (n=2)'),
        #     ('./raw_stats-360M-llmaj.json', 1, 'LLM-as-a-Judge'),
        # ]
        
        rows = [
            ('./raw_stats-smol-360M-1-4-1016.json', 1, 'n=4 participants, 1 deceptive'),
            ('./raw_stats-smol-360M-2-4-1016.json', 1, 'n=4 participants, 2 deceptive'),
            ('./raw_stats-smol-360M-3-4-1016.json', 1, 'n=4 participants, 3 deceptive'),
        ]
                
        keys = None
        
        # Make subplots
        fig, axes = plt.subplots(len(rows), num_cols+draw_regression, figsize=(40, 10*len(rows)/2))
        fig.subplots_adjust(top=0.98)
        fig.subplots_adjust(bottom=0.12)
        fig.subplots_adjust(left=0.04)
        fig.subplots_adjust(right=0.98)
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.1)
        
        keys = set()
        
        for i, row in enumerate(rows):
            
            path, num_deceptive, row_label = row
            
            with open(path, 'r') as f:
                data = json.load(f)
                
            keys = (keys & set(data[hkey].keys())) if keys else set(data[hkey].keys())
        
        for i, row in enumerate(rows):
            
            path, num_deceptive, row_label = row
            
            with open(path, 'r') as f:
                data = json.load(f)
                
            if i == 0:
                keys = sorted(list(keys), key=lambda x: np.mean(data[hkey][x][:4]))
                print(len(keys))
            
            if draw_regression:
                pdf_path = path.replace("json", "pdf").replace("raw_stats", "fit")
                pdftopng.convert(pdf_path=pdf_path, png_path=pdf_path.replace("pdf", "png"))
                image = Image.open(pdf_path.replace("pdf", "png"))
                axes[i, -1].imshow(image.crop((image.width*0.05, image.height*0.11, image.width*0.905, image.height*0.95)), aspect='auto')
                # Turn off the axis spines (the lines that frame the plot)
                axes[i, -1].spines['top'].set_visible(False)
                axes[i, -1].spines['right'].set_visible(False)
                axes[i, -1].spines['bottom'].set_visible(False)
                axes[i, -1].spines['left'].set_visible(False)

                # Turn off the ticks
                axes[i, -1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                axes[i, -1].get_xaxis().set_ticks([])
                axes[i, -1].get_yaxis().set_ticks([])
                axes[i, -1].set_ylabel('0-1 Honesty Score and Regressed Honesty Score', fontsize=10)
                axes[i, -1].set_xlabel(f'{"Peer Prediction" if i<4 else "LLM-as-a-Judge"} Score', fontsize=10)
                # axes[i, -1].set_xlim(image.width//11, image.width*10//11)
                # axes[i, -1].set_ylim(image.height//11, image.height*10//11)
                
            for j in range(num_cols):
                ax = axes[i, j]
                
                range_start = j * len(keys) // 5
                range_end = (j+1) * len(keys) // 5
                
                cur_keys = keys[range_start:range_end]
                
                means_per_key = [[], []]
                std_errs_per_key = [[], []]
                
                for key in cur_keys:
                    raw_vals = [[], []]
                    for k in range(3):
                        values = data[hkey][key][k]
                        raw_vals[int(k >= num_deceptive)] += values
                    
                    for k in range(2):
                        values = raw_vals[k]
                        means_per_key[k].append(np.mean(values))
                        std_errs_per_key[k].append(np.std(values) / np.sqrt(len(values)))
                
                for k in range(2):
                    color = ['red', 'blue'][k]
                    x_vals = [x + (k-0.5)*0.2 for x in range(range_start, range_end)]
                    ax.scatter(x_vals, means_per_key[k], label=['Deceptive', 'Honest'][k], color=color)
                    ax.errorbar(x_vals, means_per_key[k], yerr=std_errs_per_key[k], fmt='o', capsize=5, color=color, label='NA')
                    ax.fill_between(x_vals, np.array(means_per_key[k])-np.array(std_errs_per_key[k]), np.array(means_per_key[k])+np.array(std_errs_per_key[k]), color=color, alpha=0.1, label='NA')
        
                ax.set_xticks(range(range_start, range_end))
                if i == len(rows)-1:
                    ax.set_xticklabels([clean_key(key) for key in cur_keys], rotation=45, ha='right')
                else:
                    ax.set_xticklabels(['']*len(cur_keys))
                ax.grid(True)
            
            axes[i, 0].set_ylabel(f'Mean {target} in {row_label}', fontsize=10)
            
        handles, labels = [], []
        for i in range(2):
            for j in range(5):
                ax = axes[i, j]
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label in ['Deceptive', 'Honest'] and label not in labels:
                        handles.append(handle)
                        labels.append(label)
        
        fig.legend(handles[::-1], labels[::-1], loc=(0.8763, 0.0166), ncol=1, fontsize=25)
        
        plt.show()
        plt.savefig(f"compare_deceptivey_{hkey}.pdf")
        
                
def draw_scaling():
    
    means = [
        [0.59838, 0.72181, 0.80192, 0.77350], # Peer Prediction (|P|=4)
        [0.36819, 0.53812, 0.66296, None], # Peer Prediction (|P|=8)
        [0.99482, 0.76822, 0.53458, 0.32365], # LLM-as-a-Judge
    ]
    
    for i in range(3):
        for j in range(4):
            if means[i][j] is not None:
                means[i][j] = 1 - means[i][j]
    
    stderrs = [
        [0.00362, 0.00673, 0.00647, 0.00384],
        [0.02045, 0.01077, 0.01079, None],
        [0.00436, 0.00432, 0.00400, 0.00326],
    ]
    
    labels = [
        'Peer Prediction (|P|=4)',
        'Peer Prediction (|P|=8)',
        'LLM-as-a-Judge',
    ]
    
    plt.scatter(range(3), means[0][:3], label=labels[0], color='red')
    plt.errorbar(range(3), means[0][:3], yerr=stderrs[0][:3], fmt='o', capsize=5, color='red')
    plt.fill_between(range(3), np.array(means[0][:3])-np.array(stderrs[0][:3]), np.array(means[0][:3])+np.array(stderrs[0][:3]), color='red', alpha=0.1)
    # plt.scatter([3], means[0][3], color='red')
    # plt.errorbar([3], means[0][3], yerr=stderrs[0][3], fmt='o', capsize=5, color='red')
    
    plt.scatter(range(3), means[1][:3], label=labels[1], color='blue')
    plt.errorbar(range(3), means[1][:3], yerr=stderrs[1][:3], fmt='o', capsize=5, color='blue')
    plt.fill_between(range(3), np.array(means[1][:3])-np.array(stderrs[1][:3]), np.array(means[1][:3])+np.array(stderrs[1][:3]), color='blue', alpha=0.1)
    
    plt.scatter(range(4), means[2], label=labels[2], color='green')
    plt.errorbar(range(4), means[2], yerr=stderrs[2], fmt='o', capsize=5, color='green')
    plt.fill_between(range(4), np.array(means[2])-np.array(stderrs[2]), np.array(means[2])+np.array(stderrs[2]), color='green', alpha=0.1)
    
    plt.xlabel('Judge/Jury Model Size')
    plt.xticks(range(4), ['0.5B', '1.5B', '3B', '7B'])
    plt.ylabel('Punishment on Deceptive Witnesses (Higher is Better)')
    plt.legend()
    plt.show()
    plt.grid(True)
    plt.savefig("scaling.pdf")


def draw_scaling_new():
    
    # data_path = './scaling-data-celoss-logistic.json'
    # data_path = './scaling-data-mixed-celoss-logistic-indclassify-8B.json'
    data_path = './scaling-data-mixed-mislead-celoss-logistic.json'
    
    # Load the data
    with open(data_path, 'r') as f:
        # data contains a mapping from curve name to a list of tuples (y, sample size, stderr)
        all_data: Dict[str, List[Tuple[float, int, float]]] = json.load(f)
    
    key2size = {
        '8B': 8.03,
        '2B': 2.61,
        '27B': 27.2,
        '7B/8B/9B': 8.03,
    }
    
    if len(all_data) == 3:
    
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(11.5, 6), gridspec_kw={'height_ratios': [5, 3]})
        fig.subplots_adjust(left=0.1, right=0.9, top=0.97, bottom=0.08)
        
        # reduce spacing between subplots
        plt.subplots_adjust(hspace=0.1)
        plt.subplots_adjust(wspace=0.1)
        
        gs = axs[0, 0].get_gridspec()
        axs[0, 0].remove()
        axs[0, 1].remove()
        axbig = fig.add_subplot(gs[0, :])
        
        axs = {
            '8B': axbig,
            '2B': axs[1, 0],
            '27B': axs[1, 1],
        }
        
        custom_legends = []
    
    elif len(all_data) == 1:
        
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(11, 6))
        fig.subplots_adjust(left=0.1, right=0.9, top=0.97, bottom=0.08)
        
        axs = {
            list(all_data.keys())[0]: axs,
        }
        
        custom_legends = [
            # 'Peer Prediction: |P|=6, Llama 3.1 8B (deceptive/honest) + Mistral 7B v0.3 (deceptive/honest) + Gemma-2 9B (deceptive/honest)',
            'Peer Prediction: |P|=4, Llama 3.1 8B (honest/deceptive) + Mistral 7B v0.3 (honest) + Gemma-2 9B (honest)',
            'LLM-as-a-Judge: 6-shot',
            'LLM-as-a-Judge: 0-shot',
        ]
    
    fmts = ['o', '^', 's', 'v', 'D', 'P']
    legend_count = 0
    
    for fmt, content in zip(fmts, all_data.items()):
        
        key, data = content
        ax = axs[key]
    
        # All curves share the same list of x-values
        x_values = key2size[key] / np.array([0.135, 0.36, 0.494, 1.54, 3.09, 7.62])
        lnn = len(x_values)
        
        # Plot each curve
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
        for curve, color in list(zip(data.items(), colors))[::-1]:
            curve_name, curve_data = curve
            while len(curve_data) and curve_data[-1] is None:
                curve_data.pop()
            if not len(curve_data):
                continue
            y_values = []
            sample_sizes = []
            ci_low = []
            ci_high = []
            for point in curve_data:
                y_values.append(point['metric'])
                sample_sizes.append(point['n'])
                ci_low.append(point['90ci'][0])
                ci_high.append(point['90ci'][1])
            y_values = np.array(y_values)
            ci_low = np.array(ci_low)
            ci_high = np.array(ci_high)
            lnn = len(y_values)
            label = curve_name.replace(')', f' participants of size {key})').replace('|P|=','').replace('dge', f'dge on {key} participants')
            if custom_legends: label = custom_legends[legend_count]
            legend_count += 1
            ax.errorbar(x_values[:len(y_values)], y_values, yerr=(y_values-ci_low, ci_high-y_values), fmt=fmt, label=label, color=color)
            ax.plot(x_values[:len(y_values)], y_values, color=color)
            ax.fill_between(x_values[:len(y_values)], ci_low, ci_high, color=color, alpha=0.1)
    
        ax.hlines(np.log(2), x_values[0]*4, x_values[lnn-1]/4, color='black', linestyle='--', label='Random Guessing', linewidth=2)
        
        # log-scale for x
        ax.set_xscale('log')
        if key=='8B': ax.set_xlim(1, 100)
        else: ax.set_xlim(x_values[lnn-1]*0.9, x_values[0]*1.1)
        # ax.set_xticks(x_values, minor=True)
        # ax.set_xticklabels(['135M', '360M', '0.5B', '1.5B', '3B', '7B'], minor=True, fontsize=8, rotation=45)
        ax.set_yscale('linear')
        
        # horizontal grid lines
        ax.minorticks_on()
        ax.yaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='minor', linestyle=':')
    
        # Add labels and legend
        ax.legend(loc='best', fontsize=6)
        # ax.grid(True)
    
    # ax.set_title('Scaling Trends on Immunity to Deception\n', fontsize=12)
    fig.text(0.5, 0.02, 'Capability Gap: Participant Size Divided by Judge/Jury Size (135M-7B)', ha='center', fontsize=8)
    fig.text(0.03, 0.5, 'Susceptibility to Deception\n(Cross-Entropy Loss of Logistic Regression, Lower is Better, 90% CI)', va='center', rotation='vertical', fontsize=8, horizontalalignment='center', verticalalignment='center')
    
    plt.show()
    plt.savefig("scaling.pdf")

def draw_synergies_2():
    data_list =[[
            0.0,
            0.006584410893269451,
            0.01326575640290092,
            0.020185123936429704,
            0.026274381496999937,
            0.02965250739140246,
        ],
        [
            0.0,
            0.006856575455265672,
            0.015225064405225502,
            0.024105242002424275,
            0.03147203015125949,
            0.037997346164124535,
        ],
        [
            0.0,
            0.0053414016491434,
            0.015079417306211113,
            0.025096166744468604,
            0.03502475547319167,
            0.04737562606515966,
    ]]
    
    x_labels = [1, 2, 3, 4, 5, 6]
    curve_labels = ['α=-2', 'α=-1.5', 'α=-1']
    
    for i in range(3):
        plt.plot(x_labels, data_list[i], label=curve_labels[i], marker=['o', 's', 'v'][i])
    
    plt.ylabel('Amount of Surplus\n(Increase in R^2 Compared to Individual Maximum)', fontsize=8)
    
    # set left margin
    plt.subplots_adjust(left=0.2)
    
    plt.xlabel('Jury Population Size', fontsize=8)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("synergies_2.pdf")
    

def draw_synergies():
    models = [
        ('135M', 6), 
        ('360M', 5),
        ('0.5B', 4), 
        ('1.5B', 3), 
        ('3B', 2),
        ('7B', 1),
    ]
    
    with open('./scaling-data-powerset-multi-2.json', 'r') as f:
        data = json.load(f)
        data = data['8B']['Peer Prediction (|P|=4)']
    
    # num_bits = int(np.ceil(np.log2(len(data))))
    num_bits = 6
    print(len(data), num_bits)
    sum = 0
    
    categorization = defaultdict(list)
    synergy_values = []
    
    for i in range(1, len(data) + 1):
        info = data[i-1]
        constituents = []
        for j in range(num_bits):
            if i & (1 << j):
                constituents.append(data[(1<<j)-1])
        
        max_val = max([x[0] for x in constituents])
        mean_val = np.mean([x[0] for x in constituents])
        print(info[0]-max_val, len(constituents), mean_val, info[0])
        
        sum += info[0] - max_val
        categorization[len(constituents)].append(info[0] - max_val)
        synergy_values.append(info[0] - max_val)
    
    for i in range(1, num_bits+1):
        print(f'{i} bits: {np.mean(categorization[i])} {np.max(categorization[i])} {np.min(categorization[i])}')
    print(sum)

    return
    

    # Parameters
    m = num_bits  # Number of entities
    # entities = [f"Entity {i+1}" for i in range(m)]  # Placeholder entity names
    entities = [models[i][0] for i in range(m)][::-1]
    assert len(entities) == m

    # Generate synergy values
    num_combinations = 2**m - 1
    # synergy_values = deepcopy([data[i][0] for i in range(num_combinations)])

    # Convert integer to binary string representing entity combinations
    combinations = [list(map(int, f"{i:b}".zfill(m))) for i in range(1, num_combinations + 1)]
    print(combinations)

    # Create a DataFrame for the UpSet plot
    df_combinations = pd.DataFrame(combinations, columns=entities)
    df_combinations['Synergy'] = synergy_values

    # # Plotting Chord Diagram (simple version with synergies between pairs)
    # import networkx as nx

    # # For the Chord Diagram, consider only pairwise synergies (first mC2 values)
    # pairwise_indices = [i for i in range(1, num_combinations + 1) if bin(i).count('1') == 2]
    # pairwise_synergies = synergy_values[:len(pairwise_indices)]

    # # Create a graph for pairwise combinations
    # G = nx.Graph()
    # for idx, val in zip(pairwise_indices, pairwise_synergies):
    #     included_entities = [entities[i] for i, b in enumerate(f"{idx:b}".zfill(m)) if b == '1']
    #     G.add_edge(included_entities[0], included_entities[1], weight=val)

    # # Plot the Chord Diagram (network graph version)
    # plt.figure(figsize=(8, 8))
    # pos = nx.circular_layout(G)
    # edges = G.edges(data=True)

    # # Draw the network
    # nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
    # nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx_edges(G, pos, width=[10 * G[u][v]['weight'] for u, v in G.edges()], alpha=0.6)

    # # Show the plot
    # plt.title("Chord Diagram (Pairwise Synergies)")
    # plt.show()
    # plt.savefig("chord_diagram.pdf")

    # UpSet plot
    df_upset = df_combinations.drop(columns='Synergy').astype(bool)
    df_upset['Synergy'] = synergy_values

    # Create and show the UpSet plot
    upset = UpSet(df_upset.set_index(entities), subset_size='count', show_counts=True, sort_by="degree")
    counts = generate_counts(n_samples=100000, n_categories=m).rename_axis(entities)
    # counts = counts[1:]
    for key in counts.keys():
        counts[key] = 0
    for i in range(len(synergy_values)):
        a, b, c, d, e, f = combinations[i]
        print(a, b, c, d, e, f, synergy_values[i])
        counts.loc[(bool(a), bool(b), bool(c), bool(d), bool(e), bool(f))] = synergy_values[i]
    assert len(counts) == len(synergy_values) + 1
    print(len(counts), counts)
    # counts[1:, -1] = synergy_values
    UpSet(counts, sort_by="cardinality", min_degree=3).plot()
    plt.suptitle('UpSet Plot of Entity Synergies')
    plt.show()
    plt.savefig("upset_plot.pdf")
    
    pairwise_indices = [i-1 for i in range(1, num_combinations + 1) if bin(i).count('1') == 2]
    pairwise_synergies = [synergy_values[i] for i in pairwise_indices]

    # Create pairs for the Chord Diagram
    pairs = []
    max_val = max([abs(x) for x in synergy_values])
    for idx, val in zip(pairwise_indices, pairwise_synergies):
        # assert val >= -1e-6
        included_entities = [entities[i] for i, b in enumerate(f"{idx+1:b}".zfill(m)) if b == '1']
        pairs.append([
            included_entities[0], 
            included_entities[1], 
            (abs(val) / max_val),
            'orange' if val > 0 else 'blue',
            (abs(val) / max_val)
        ])

    # Create DataFrame for d3blocks
    df_pairs = pd.DataFrame(pairs, columns=["source", "target", "weight", "color", "opacity"])

    # Create the Chord Diagram
    d3 = D3Blocks(chart="chord")
    d3.set_node_properties(df_pairs, opacity=0.2, cmap='tab20')
    d3.set_edge_properties(df_pairs, color=None, opacity=None)
    d3.show(filepath='chord_diagram.html')
    # d3.set_node_properties(df=df_pairs, color='#ffffff', opacity=0.8)
    # d3.set_edge_properties(df=df_pairs, color='#ffffff', opacity=[pairs[i][2] for i in range(len(pairs))])
    # d3.chord(df_pairs, title='Interactive Chord Diagram of Synergies', filepath='./chord_diagram.html', color=df_pairs['color'], opacity=df_pairs['opacity'])  
    
    # Create a dictionary to map pairwise relationships
    links = []
    pairwise_indices = [i for i in range(1, num_combinations + 1) if bin(i).count('1') == 2]
    pairwise_synergies = synergy_values[:len(pairwise_indices)]

    # Create links for the Chord Diagram
    for idx, val in zip(pairwise_indices, pairwise_synergies):
        included_entities = [entities[i] for i, b in enumerate(f"{idx:b}".zfill(m)) if b == '1']
        links.append((included_entities[0], included_entities[1], val))

    # Create the Chord Diagram using HoloViews
    chord_data = hv.Dataset(links, ['source', 'target'], 'value')
    chord = hv.Chord(chord_data).opts(width=600, height=600, cmap='Category20', edge_color='value')

    # Display the plot
    # hv.output(chord)
    hv.save(chord, 'chord_diagram.png')

def draw_distribution_plot():
    from scipy.stats import gaussian_kde
    
    with open("raw_stats.json", 'r') as f:
        input_data = json.load(f)
        num_vars = 6
        var_names = [
            'Gemma-2 9B (deceptive)', 
            'Gemma-2 9B (honest)', 
            'Mistral 7B v0.3 (deceptive)', 
            'Mistral 7B v0.3 (honest)', 
            'Llama 3.1 8B (deceptive)', 
            'Llama 3.1 8B (honest)',
        ]

    # Example data: replace these with your actual lists of real-valued samples
    data = {var_names[i]: [] for i in range(num_vars)}
    for category in input_data['values_for_each']:
        for i in range(num_vars):
            data[var_names[i]].extend(input_data['values_for_each'][category][i])
    
    reduced_data = {
        'Deceptive': data[var_names[0]] + data[var_names[2]] + data[var_names[4]],
        'Honest': data[var_names[1]] + data[var_names[3]] + data[var_names[5]],
    }
    
    print({var: np.mean(data[var]) for var in data})
    print({reduced_var: np.mean(reduced_data[reduced_var]) for reduced_var in reduced_data})

    # Unified plot for all variables; compact margins
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    for variable, samples in data.items():
        if not samples:
            continue
        print(np.median(samples), np.mean(samples), np.std(samples), np.min(samples), np.max(samples))
        samples = np.array(samples) + np.random.normal(0, 0.5, len(samples))
        kde = gaussian_kde(samples, bw_method='scott')  # KDE with Scott's bandwidth
        x_range = np.linspace(samples.min() - 1, samples.max() + 1, 500)  # Range for the x-axis
        plt.plot(x_range, kde(x_range), label=variable, linewidth=2)
        plt.fill_between(x_range, kde(x_range), alpha=0.2)

    # plt.yscale('log')
    plt.title('Distributions of Scores for Different Models')
    plt.xlabel('Score')
    plt.ylabel('Density')
    # plt.ylim(1e-3, 1)
    plt.xlim(-15, 15)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()
    
    plt.savefig("distribution_plot.pdf")


# draw_synergies_2()
# compare_deceptive()
# draw_scaling_new()
# draw_distribution_plot()