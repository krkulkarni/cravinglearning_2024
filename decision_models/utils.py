import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# import arviz as az
import math
import pickle
import datetime

from pyEM.math import compGauss_ms, norm2alpha, norm2beta

def logical_xor(a, b):
    if bool(a) == bool(b):
        return 0
    else:
        return a or b
    
def plot_ppc(batch, pid_num, block, pid_list):
    b = 0 if block=='money' else 1
    n_rows = np.max([int(np.ceil(len(batch)/3))-1, 1])
    fig, ax = plt.subplots(ncols=3, nrows=n_rows, figsize=(15,n_rows*2), facecolor='w', edgecolor='k')
    # Flatten axes
    ax = ax.flatten()
    for m, (model_name, model) in enumerate(batch.items()):
        if model_name == 'Group':
            continue
        pred_actions = model.predicted_actions[b, pid_num, :, :]
        for i, row in enumerate(pred_actions):
            sns.scatterplot(x=np.arange(len(row)), y=row, color='blue', alpha=0.01, ax=ax[m])
            if i > 50:
                break
        # az.plot_hdi(x=np.arange(posterior_pred.shape[1]), y=posterior_pred, hdi_prob=0.60, ax=ax[m])
        pid_act = model.longform[(model.longform['PID']==pid_list[pid_num]) & (model.longform['Type']==block)]['Action'].values
        sns.scatterplot(x=np.arange(len(pid_act)), y=pid_act+0.1, color='red', ax=ax[m])
        ax[m].set_title(f'{model.short_name}')
    fig.suptitle(f'{batch["Group"].capitalize()} - {block.capitalize()}')
    plt.tight_layout()
    plt.close()
    return fig

def model_comparison(modout, group, skip_models=[], ic_type='bic'):
    full_money = pd.DataFrame()
    full_other = pd.DataFrame()
    for model_name, model in modout.items():
        if model_name[:-6] in skip_models:
            continue
        best_fit = model['fit'][ic_type][:, 0]
        # print(f'{model_name} - index: -2 - {np.sum(model["fit"][ic_type][:, -2])}')
        # print(f'{model_name} - index: -1 - {np.sum(best_fit)}')
        # print(f'{model_name} - index: 0 - {np.sum(model["fit"][ic_type][:, 0])}')
        # print(f'{model_name} - index: 1 - {np.sum(model["fit"][ic_type][:, 1])}')
        # Replace nan with mean value
        best_fit[np.isnan(best_fit)] = np.nanmean(best_fit)
        summed_ic = np.sum(best_fit)
        if 'money' in model_name:
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': ['Money'],
                'IC': [summed_ic]
            })
            full_money = pd.concat([full_money, model_df], ignore_index=True)
        elif 'other' in model_name:
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': ['Other'],
                'IC': [summed_ic]
            })
            full_other = pd.concat([full_other, model_df], ignore_index=True)
    full_money['Delta-IC'] = full_money['IC'] - full_money['IC'].min()
    full_other['Delta-IC'] = full_other['IC'] - full_other['IC'].min()
    # full_money['Delta-IC'] = full_money['IC']
    # full_other['Delta-IC'] = full_other['IC']

    fig, ax = plt.subplots(figsize=(10,5), ncols=2, facecolor='w')
    plots = sns.barplot(data=full_money.sort_values('Model'), y='Model', x='Delta-IC', color='cornflowerblue', ax=ax[0])
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='left', va='center')
    ax[0].set_title('Money Cue')
    plots = sns.barplot(data=full_other.sort_values('Model'), y='Model', x='Delta-IC', color='darkorange', ax=ax[1])
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='left', va='center')
    ax[1].set_title('Addictive Cue')
    fig.suptitle(f'{group.capitalize()} Delta-IC')
    plt.tight_layout()
    plt.close()
    return fig

def model_comparison_for_paper(modout, group, skip_models=[], ic_type='bic', figsize=(5,3)):
    full_money = pd.DataFrame()
    full_other = pd.DataFrame()

    if group == 'alcohol':
        barcolor = sns.palettes.color_palette('Set2')[2]
    elif group == 'cannabis':
        barcolor = sns.palettes.color_palette('Set2')[5]

    for model_name, model in modout.items():
        if model_name[:-6] in skip_models:
            continue
        best_fit = model['fit'][ic_type][:, 0]
        # print(f'{model_name} - index: -2 - {np.sum(model["fit"][ic_type][:, -2])}')
        # print(f'{model_name} - index: -1 - {np.sum(best_fit)}')
        # print(f'{model_name} - index: 0 - {np.sum(model["fit"][ic_type][:, 0])}')
        # print(f'{model_name} - index: 1 - {np.sum(model["fit"][ic_type][:, 1])}')
        # Replace nan with mean value
        best_fit[np.isnan(best_fit)] = np.nanmean(best_fit)
        summed_ic = np.sum(best_fit)
        if 'money' in model_name:
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': ['Money'],
                'IC': [summed_ic]
            })
            full_money = pd.concat([full_money, model_df], ignore_index=True)
        elif 'other' in model_name:
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': ['Other'],
                'IC': [summed_ic]
            })
            full_other = pd.concat([full_other, model_df], ignore_index=True)
    full_money['Delta-IC'] = full_money['IC'] - full_money['IC'].min()
    full_other['Delta-IC'] = full_other['IC'] - full_other['IC'].min()
    # full_money['Delta-IC'] = full_money['IC']
    # full_other['Delta-IC'] = full_other['IC']

    fig, ax = plt.subplots(figsize=figsize, ncols=1, facecolor='w')
    plots = sns.barplot(data=full_money.sort_values('Model'), y='Model', x='Delta-IC', color=barcolor, ax=ax, alpha=0.5)
    plots = sns.barplot(data=full_money.sort_values('Model'), y='Model', x='Delta-IC', fill=False, edgecolor='black', ax=ax)
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        if w>175:
            plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='center', va='center', color='black', fontsize=20)
        else:
            plots.annotate(f'+{np.round(w,2)}', (w+5, h/2+y), ha='left', va='center', color='black', fontsize=20)
    # ax.set_title('Money Cue')
    ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.setp(ax.spines.values(), linewidth=2)
    plt.setp(ax.patches, linewidth=2)
    sns.despine()
    plt.tight_layout()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=figsize, ncols=1, facecolor='w')
    plots = sns.barplot(data=full_other.sort_values('Model'), y='Model', x='Delta-IC', color=barcolor, ax=ax2, alpha=0.5)
    plots = sns.barplot(data=full_other.sort_values('Model'), y='Model', x='Delta-IC', fill=False, edgecolor='black', ax=ax2)
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        if w>175:
            plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='center', va='center', color='black', fontsize=20)
        else:
            plots.annotate(f'+{np.round(w,2)}', (w+5, h/2+y), ha='left', va='center', color='black', fontsize=20)
    # ax.set_title('Addictive Cue')
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    plt.setp(ax2.spines.values(), linewidth=2)
    plt.setp(ax2.patches, linewidth=2)
    sns.despine()
    plt.tight_layout()
    plt.close()
    return fig, fig2, full_money, full_other

def model_comparison_by_group(modout, block, ic_type='bic'):
    
    full_df = pd.DataFrame()
    for group, group_modout in modout.items():
        group_df = pd.DataFrame()
        for model_name, model in group_modout.items():
            if block not in model_name:
                continue
            best_fit = model['fit'][ic_type][:, 0]
            best_fit[np.isnan(best_fit)] = np.nanmean(best_fit)
            summed_ic = np.sum(best_fit)
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': [block.capitalize()],
                'IC': [summed_ic],
                'Group': [group]
            })
            group_df = pd.concat([group_df, model_df], ignore_index=True)
        group_df['Delta-IC'] = group_df['IC'] - group_df['IC'].min()
        full_df = pd.concat([full_df, group_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10,5), ncols=len(modout.keys()), facecolor='w')
    for i, group in enumerate(modout.keys()):
        plots = sns.barplot(data=full_df[full_df['Group']==group].sort_values('Model'), y='Model', x='Delta-IC', color='gray', ax=ax[i])
        for bar in plots.patches:
            w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
            plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='left', va='center')
        ax[i].set_title(group.capitalize())

    # plots = sns.barplot(data=full_money.sort_values('Model'), y='Model', x='Delta-IC', color='cornflowerblue', ax=ax[0])
    # for bar in plots.patches:
    #     w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
    #     plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='left', va='center')
    # ax[0].set_title('Money Cue')
    # plots = sns.barplot(data=full_other.sort_values('Model'), y='Model', x='Delta-IC', color='darkorange', ax=ax[1])
    # for bar in plots.patches:
    #     w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
    #     plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='left', va='center')
    # ax[1].set_title('Addictive Cue')
    if block == 'money':
        fig.suptitle('Money Cue Delta-IC')
    elif block == 'other':
        fig.suptitle('Addictive Cue Delta-IC')
    plt.tight_layout()
    plt.close()
    return fig

def craving_model_comparison_by_group_for_paper(modout, block, ic_type='bic', figsize=(5,3)):
    
    full_df = pd.DataFrame()
    for group, group_modout in modout.items():
        group_df = pd.DataFrame()
        for model_name, model in group_modout.items():
            if block not in model_name:
                continue
            best_fit = model['fit'][ic_type][:, 0]
            best_fit[np.isnan(best_fit)] = np.nanmean(best_fit)
            summed_ic = np.sum(best_fit)
            model_df = pd.DataFrame({
                'Model': [model_name[:-6]],
                'Type': [block.capitalize()],
                'IC': [summed_ic],
                'Group': [group]
            })
            group_df = pd.concat([group_df, model_df], ignore_index=True)
        group_df['Delta-IC'] = group_df['IC'] - group_df['IC'].min()
        full_df = pd.concat([full_df, group_df], ignore_index=True)

    # for i, group in enumerate(modout.keys()):
    #     fig, ax = plt.subplots(figsize=figsize, ncols=1, facecolor='w')
    #     if group =='alcohol':
    #         barcolor = sns.palettes.color_palette('Set2')[2]
    #     elif group == 'cannabis':
    #         barcolor = sns.palettes.color_palette('Set2')[5]
    #     plots = sns.barplot(data=full_df[full_df['Group']==group].sort_values('Model'), y='Model', x='Delta-IC', color=barcolor, ax=ax, alpha=0.5)
    #     plots = sns.barplot(data=full_df[full_df['Group']==group].sort_values('Model'), y='Model', x='Delta-IC', fill=False, edgecolor='black', ax=ax)
    #     for bar in plots.patches:
    #         w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
    #         if w>175:
    #             plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='center', va='center')
    #         else:
    #             plots.annotate(f'+{np.round(w,2)}', (w+20, h/2+y), ha='left', va='center')
    #     ax.set_title('')
    #     ax.set_yticklabels([])
    #     ax.set_ylabel('')
    #     ax.set_xlabel('')
    #     plt.setp(ax.spines.values(), linewidth=5)

    # sns.despine()
    
    # plt.tight_layout()
    # plt.close()
    # return fig

    fig, ax = plt.subplots(figsize=figsize, ncols=1, facecolor='w')
    plots = sns.barplot(data=full_df[full_df['Group']=='alcohol'], y='Model', x='Delta-IC', color=sns.palettes.color_palette('Set2')[2], ax=ax, alpha=0.5)
    plots = sns.barplot(data=full_df[full_df['Group']=='alcohol'], y='Model', x='Delta-IC', fill=False, edgecolor='black', ax=ax)
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        if w>175:
            plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='center', va='center', color='black', fontsize=20)
        else:
            plots.annotate(f'+{np.round(w,2)}', (w+5, h/2+y), ha='left', va='center', color='black', fontsize=20)
    # ax.set_title('Money Cue')
    ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.setp(ax.spines.values(), linewidth=2)
    plt.setp(ax.patches, linewidth=2)
    sns.despine()
    plt.tight_layout()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=figsize, ncols=1, facecolor='w')
    plots = sns.barplot(data=full_df[full_df['Group']=='cannabis'], y='Model', x='Delta-IC', color=sns.palettes.color_palette('Set2')[5], ax=ax2, alpha=0.5)
    plots = sns.barplot(data=full_df[full_df['Group']=='cannabis'], y='Model', x='Delta-IC', fill=False, edgecolor='black', ax=ax2)
    for bar in plots.patches:
        w, h, y = bar.get_width(), bar.get_height(), bar.get_y()
        if w>175:
            plots.annotate(f'+{np.round(w,2)}', (w/2, h/2+y), ha='center', va='center', color='black', fontsize=20)
        else:
            plots.annotate(f'+{np.round(w,2)}', (w+5, h/2+y), ha='left', va='center', color='black', fontsize=20)
    # ax.set_title('Addictive Cue')
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    plt.setp(ax2.spines.values(), linewidth=2)
    plt.setp(ax2.patches, linewidth=2)
    sns.despine()
    plt.tight_layout()
    plt.close()
    return fig, fig2, full_df

def parameter_recovery_plot(model, sim_model):
    fig, ax = plt.subplots(1, len(model.parnames), figsize=(15, 5))
    for b, block in enumerate(['Money', 'Other']):
        block_table = model.table[model.table['Type']==block]
        sim_block_table = sim_model.table[sim_model.table['Type']==block]
        for i, var in enumerate(model.parnames):
            sns.regplot(
                x=block_table[var].values,
                y=sim_block_table[var].values,
                ax=ax[i]
            )
            # Annotate with Pearson's r
            r, p = pearsonr(block_table[var].values, sim_block_table[var].values)
            ax[i].annotate(
                f'{block.capitalize()}: r = {r:.3f}, p = {p:.3f}', 
                xy=(0.05, 0.95-b*0.05), 
                xycoords='axes fraction'
            )
        ax[i].set_xlabel('True')
        ax[i].set_ylabel('Recovered')
        ax[i].set_title(var)
    plt.tight_layout()
    plt.show()

def plot_craving_param_dist(model, typ='diff'):
    nrows = math.ceil(model.npars/3)
    fig, ax = plt.subplots(figsize=(15, 1+nrows*3), ncols=3, nrows=nrows)
    ax = ax.flatten()

    money_table = model.table[model.table['Type']=='Money']
    other_table = model.table[model.table['Type']=='Other']
    diff_table = pd.DataFrame({
        'PID': money_table['PID'],
        'Type': ['Diff']*len(money_table),
        # 'craving_baseline': other_table['craving_baseline'] - money_table['craving_baseline'],
        # 'outcome_weight': other_table['outcome_weight'] - money_table['outcome_weight'],
        # 'ev_weight': other_table['ev_weight'] - money_table['ev_weight'],
        # 'rpe_weight': other_table['rpe_weight'] - money_table['rpe_weight'],
        # 'gamma': other_table['gamma'] - money_table['gamma'],
        # 'craving_sd': other_table['craving_sd'] - money_table['craving_sd']
    })
    for var in model.parnames:
        diff_table[var] = other_table[var] - money_table[var]

    big_table = pd.concat([money_table, other_table, diff_table], ignore_index=True)

    if typ == 'diff':
        chosen_table = big_table
        len_line = 2.45
    else:
        chosen_table = model.table
        len_line = 1.45
    for i, var in enumerate(model.parnames):
        # sns.distplot(block_result_df[var], ax=ax.flatten()[list(block_result_df.columns).index(var)])
        # sns.stripplot(block_table[var], ax=ax[i])
        sns.stripplot(data=chosen_table, x='Type', y=var, hue='Type', ax=ax[i])
        sns.boxplot(data=chosen_table, x='Type', y=var, ax=ax[i], showfliers=False, boxprops=dict(alpha=.3))
        ax[i].hlines(0, -0.45, len_line, linestyle='--', color='black')
        # find 75th percentile of block_result_df[var]
        bottom = np.percentile(chosen_table[var], 5)
        top = np.percentile(chosen_table[var], 95)
        if var not in ['craving_sd', 'LL']:
            ax[i].set_ylim(bottom, top)
        ax[i].set_title(var)
        ax[i].set_xticks([])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
    plt.show()

def plot_mood_param_dist(model, typ='diff'):
    nrows = math.ceil(model.npars/3)
    fig, ax = plt.subplots(figsize=(15, 1+nrows*3), ncols=3, nrows=nrows)
    ax = ax.flatten()

    money_table = model.table[model.table['Type']=='Money']
    other_table = model.table[model.table['Type']=='Other']
    diff_table = pd.DataFrame({
        'PID': money_table['PID'],
        'Type': ['Diff']*len(money_table),
        # 'mood_baseline': other_table['mood_baseline'] - money_table['mood_baseline'],
        # 'outcome_weight': other_table['outcome_weight'] - money_table['outcome_weight'],
        # 'ev_weight': other_table['ev_weight'] - money_table['ev_weight'],
        # 'rpe_weight': other_table['rpe_weight'] - money_table['rpe_weight'],
        # 'gamma': other_table['gamma'] - money_table['gamma'],
        # 'mood_sd': other_table['mood_sd'] - money_table['mood_sd']
    })
    for var in model.parnames:
        diff_table[var] = other_table[var] - money_table[var]

    big_table = pd.concat([money_table, other_table, diff_table], ignore_index=True)

    if typ == 'diff':
        chosen_table = big_table
        len_line = 2.45
    else:
        chosen_table = model.table
        len_line = 1.45
    for i, var in enumerate(model.parnames):
        # sns.distplot(block_result_df[var], ax=ax.flatten()[list(block_result_df.columns).index(var)])
        # sns.stripplot(block_table[var], ax=ax[i])
        sns.stripplot(data=chosen_table, x='Type', y=var, hue='Type', ax=ax[i])
        sns.boxplot(data=chosen_table, x='Type', y=var, ax=ax[i], showfliers=False, boxprops=dict(alpha=.3))
        ax[i].hlines(0, -0.45, len_line, linestyle='--', color='black')
        # find 75th percentile of block_result_df[var]
        bottom = np.percentile(chosen_table[var], 5)
        top = np.percentile(chosen_table[var], 95)
        if var not in ['mood_sd', 'LL']:
            ax[i].set_ylim(bottom, top)
        ax[i].set_title(var)
        ax[i].set_xticks([])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
    plt.show()

# def plot_sim_craving(pid_num, block_num, model_dict, n_sims=1000):

#     nrows = math.ceil(len(model_dict)/3)
#     ncols = 3
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 1+nrows*2))
#     ax = ax.flatten()

#     for m, (model_name, model) in enumerate(model_dict.items()):
#         sim_pred_cravings = np.zeros((2, 1000, model.num_craving_trials))
#         for i in range(n_sims):
#             sim_pred_cravings[block_num, i, :] = np.random.normal(loc=model.pred_cravings[block_num, pid_num], scale=np.exp(model.param_est[block_num, -1, pid_num]))
        
#         az.plot_hdi(range(model.num_craving_trials), sim_pred_cravings[block_num], hdi_prob=0.95, color='k', ax=ax[m], fill_kwargs={'alpha': 0.2})
#         az.plot_hdi(range(model.num_craving_trials), sim_pred_cravings[block_num], hdi_prob=0.50, color='k', ax=ax[m], fill_kwargs={'alpha': 0.5})
#         az.plot_hdi(range(model.num_craving_trials), sim_pred_cravings[block_num], hdi_prob=0.025, color='k', ax=ax[m], fill_kwargs={'alpha': 0.9})
#         sns.lineplot(x=range(model.num_craving_trials), y=model.true_cravings[block_num, pid_num, :], ax=ax[m], color='blue')
#         ax[m].set_xlabel('Trial')
#         ax[m].set_ylabel('Craving')
#         block_names= ['Money', 'Other']
#         ax[m].set_title(f'Model {model_name} - PID {pid_num} - {block_names[block_num]}')
#     plt.tight_layout()
#     plt.show()

# def plot_sim_mood(pid_num, block_num, model_dict, n_sims=1000):

#     fig, ax = plt.subplots(len(model_dict.keys())//3, 3, figsize=(15, 3+len(model_dict.keys())//3))
#     ax = ax.flatten()

#     for m, (model_name, model) in enumerate(model_dict.items()):
#         sim_pred_moods = np.zeros((2, 1000, model.num_mood_trials))
#         for i in range(n_sims):
#             sim_pred_moods[block_num, i, :] = np.random.normal(loc=model.pred_moods[block_num, pid_num], scale=np.exp(model.param_est[block_num, -1, pid_num]))
        
#         az.plot_hdi(range(model.num_mood_trials), sim_pred_moods[block_num], hdi_prob=0.95, color='k', ax=ax[m], fill_kwargs={'alpha': 0.2})
#         az.plot_hdi(range(model.num_mood_trials), sim_pred_moods[block_num], hdi_prob=0.50, color='k', ax=ax[m], fill_kwargs={'alpha': 0.5})
#         az.plot_hdi(range(model.num_mood_trials), sim_pred_moods[block_num], hdi_prob=0.025, color='k', ax=ax[m], fill_kwargs={'alpha': 0.9})
#         sns.lineplot(x=range(model.num_mood_trials), y=model.true_moods[block_num, pid_num, :], ax=ax[m], color='blue')
#         ax[m].set_xlabel('Trial')
#         ax[m].set_ylabel('Craving')
#         block_names= ['Money', 'Other']
#         ax[m].set_title(f'Model {model_name} - PID {pid_num} - {block_names[block_num]}')
#     plt.tight_layout()
#     plt.show()

def store_modout(
        modout,
        model_name, param_names, subj_dict, fit_func,
        m, inv_h, posterior, NPL, NLPrior, NLL,
        save_fit_path=None
    ):

    nsubjects = len(subj_dict['choices'])
    ntrials = 60
    nblocks = 1

    # get covariance matrix
    _, _, _, covmat_out = compGauss_ms(m,inv_h,2)
    nparams = len(param_names)

    # Fill in general information
    modout[model_name] = {}
    modout[model_name]['date'] = datetime.date.today().strftime('%Y%m%d')
    modout[model_name]['behavior'] = subj_dict  # copy behavior here

    # Fill in fit information
    est_params = m.T.copy()
    for subj_idx in range(nsubjects):
        for param_idx, param_name in enumerate(param_names):
            if 'beta' in param_name:
                est_params[subj_idx, param_idx] = norm2beta(m[param_idx, subj_idx])
            elif 'lr' in param_name:
                est_params[subj_idx, param_idx] = norm2alpha(m[param_idx, subj_idx])

    modout[model_name]['fit'] = {}
    modout[model_name]['fit']['norm_params'] = m
    modout[model_name]['fit']['params'] = est_params
    modout[model_name]['fit']['param_names'] = param_names
    modout[model_name]['fit']['inverse_hess'] = inv_h
    modout[model_name]['fit']['gauss.mu'] = posterior['mu']
    modout[model_name]['fit']['gauss.sigma'] = posterior['sigma']
    modout[model_name]['fit']['gauss.cov'] = covmat_out
    try:
        modout[model_name]['fit']['gauss.corr'] = np.corrcoef(covmat_out)
    except:
        print('covariance mat not square, symmetric, or positive semi-definite')
        modout[model_name]['fit']['gauss.corr'] = np.eye(nparams)
    modout[model_name]['fit']['npl'] = NPL  # note: this is the negative joint posterior likelihood
    modout[model_name]['fit']['NLPrior'] = NLPrior
    modout[model_name]['fit']['nll'] = NPL - NLPrior
    modout[model_name]['fit']['aic'] = 2*nparams + 2*modout[model_name]['fit']['nll']
    modout[model_name]['fit']['bic'] = np.log(ntrials*nblocks)*nparams + 2*modout[model_name]['fit']['nll']
    modout[model_name]['fit']['lme'] = []

    # Make sure you know if BIC is positive or negative! and replace lme with
    # bic if covariance is negative.
    # Error check that BICs are in a similar range
    # Get subject specifics
    goodHessian = np.zeros(nsubjects)
    modout[model_name]['fit']['ev']          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    modout[model_name]['fit']['ch_prob']     = np.zeros((nsubjects, nblocks, ntrials,   2))
    modout[model_name]['fit']['choices']     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    modout[model_name]['fit']['choices_A']   = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['rewards']     = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['pe']          = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['negll']       = np.zeros((nsubjects,))

    # from mfit_optimize_hierarchical.m from Sam Gershman
    # Also reference Daw 2009 (Equation 17) for Laplace approximation
    for subj_idx in range(nsubjects):
        try:
            det_inv_hessian = np.linalg.det(inv_h[:, :, subj_idx])
            hHere = np.linalg.slogdet(inv_h[:, :, subj_idx])[1]
            L = -NPL - 0.5 * np.log(1 / det_inv_hessian) + (nparams / 2) * np.log(2 * np.pi)
            goodHessian[subj_idx] = 1
        except:
            print('Hessian is not positive definite')
            try:
                hHere = np.linalg.slogdet(inv_h[:,:,subj_idx])[1]
                L = np.nan
                goodHessian[subj_idx] = 0
            except:
                print('could not calculate')
                goodHessian[subj_idx] = -1
                L = np.nan
        modout[model_name]['fit']['lme'] = L
        modout[model_name]['fit']['goodHessian'] = goodHessian

        # Get subjectwise model predictions# get info for current subject
        choices = np.array(subj_dict['choices'])[subj_idx,:,:]
        rewards = np.array(subj_dict['rewards'])[subj_idx,:,:]
        craving_ratings = np.array(subj_dict['craving_ratings'])[subj_idx,:,:]
        
        model_fits = fit_func(m[:,subj_idx], choices, rewards, craving_ratings, prior=None, output='all')
        # print(model_fits)

        # store model fits
        modout[model_name]['fit']['ev'][subj_idx,:,:,:] = model_fits['ev']
        modout[model_name]['fit']['ch_prob'][subj_idx,:,:,:] = model_fits['ch_prob']
        modout[model_name]['fit']['choices'][subj_idx,:,:] = model_fits['choices']
        modout[model_name]['fit']['choices_A'][subj_idx,:,:] = model_fits['choices_A']
        modout[model_name]['fit']['rewards'][subj_idx,:,:] = model_fits['rewards']
        modout[model_name]['fit']['pe'][subj_idx,:,:] = model_fits['pe']
        modout[model_name]['fit']['negll'][subj_idx] = model_fits['negll']

    # # Save output
    # if save_fit_path is not None:
    #     with open(f'{save_fit_path}/EMfit_{model_name}.pkl', 'wb') as f:
    #         pickle.dump(modout, f)
    
    return modout

def store_craving_modout(
        modout,
        model_name, param_names, all_data, fit_func,
        m, inv_h, posterior, NPL, NLPrior, NLL,
        save_fit_path=None
    ):

    nsubjects = len(all_data)
    ntrials = 60
    nblocks = 1

    # get covariance matrix
    _, _, _, covmat_out = compGauss_ms(m,inv_h,2)
    nparams = len(param_names)

    # Fill in general information
    modout[model_name] = {}
    modout[model_name]['date'] = datetime.date.today().strftime('%Y%m%d')
    # modout[model_name]['behavior'] = subj_dict  # copy behavior here

    # Fill in fit information
    est_params = m.T.copy()
    for subj_idx in range(nsubjects):
        for param_idx, param_name in enumerate(param_names):
            # if 'beta' in param_name:
            #     est_params[subj_idx, param_idx] = norm2beta(m[param_idx, subj_idx])
            if 'gamma' in param_name:
                est_params[subj_idx, param_idx] = norm2alpha(m[param_idx, subj_idx])

    modout[model_name]['fit'] = {}
    modout[model_name]['fit']['norm_params'] = m
    modout[model_name]['fit']['params'] = est_params
    modout[model_name]['fit']['param_names'] = param_names
    modout[model_name]['fit']['inverse_hess'] = inv_h
    modout[model_name]['fit']['gauss.mu'] = posterior['mu']
    modout[model_name]['fit']['gauss.sigma'] = posterior['sigma']
    modout[model_name]['fit']['gauss.cov'] = covmat_out
    try:
        modout[model_name]['fit']['gauss.corr'] = np.corrcoef(covmat_out)
    except:
        print('covariance mat not square, symmetric, or positive semi-definite')
        modout[model_name]['fit']['gauss.corr'] = np.eye(nparams)
    modout[model_name]['fit']['npl'] = NPL  # note: this is the negative joint posterior likelihood
    modout[model_name]['fit']['NLPrior'] = NLPrior
    modout[model_name]['fit']['nll'] = NPL - NLPrior
    modout[model_name]['fit']['aic'] = 2*nparams + 2*modout[model_name]['fit']['nll']
    modout[model_name]['fit']['bic'] = np.log(ntrials*nblocks)*nparams + 2*modout[model_name]['fit']['nll']
    modout[model_name]['fit']['lme'] = []

    # Make sure you know if BIC is positive or negative! and replace lme with
    # bic if covariance is negative.
    # Error check that BICs are in a similar range
    # Get subject specifics
    goodHessian = np.zeros(nsubjects)
    modout[model_name]['fit']['ev']          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    modout[model_name]['fit']['actions']     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    modout[model_name]['fit']['outcomes']     = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['rpes']          = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['craving_ratings'] = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['pred_cravings'] = np.zeros((nsubjects, nblocks, ntrials,))
    modout[model_name]['fit']['negll']       = np.zeros((nsubjects,))

    # from mfit_optimize_hierarchical.m from Sam Gershman
    # Also reference Daw 2009 (Equation 17) for Laplace approximation
    for subj_idx in range(nsubjects):
        try:
            det_inv_hessian = np.linalg.det(inv_h[:, :, subj_idx])
            hHere = np.linalg.slogdet(inv_h[:, :, subj_idx])[1]
            L = -NPL - 0.5 * np.log(1 / det_inv_hessian) + (nparams / 2) * np.log(2 * np.pi)
            goodHessian[subj_idx] = 1
        except:
            print('Hessian is not positive definite')
            try:
                hHere = np.linalg.slogdet(inv_h[:,:,subj_idx])[1]
                L = np.nan
                goodHessian[subj_idx] = 0
            except:
                print('could not calculate')
                goodHessian[subj_idx] = -1
                L = np.nan
        modout[model_name]['fit']['lme'] = L
        modout[model_name]['fit']['goodHessian'] = goodHessian

        # Get subjectwise model predictions# get info for current subject
        craving_ratings = all_data[subj_idx][0]
        actions = all_data[subj_idx][1]
        outcomes = all_data[subj_idx][2]
        evs = all_data[subj_idx][3]
        rpes = all_data[subj_idx][4]
        
        model_fits = fit_func(m[:,subj_idx], craving_ratings, actions, outcomes, evs, rpes, prior=None, output='all')

        # store model fits
        modout[model_name]['fit']['ev'][subj_idx,:,:,:] = model_fits['ev']
        modout[model_name]['fit']['actions'][subj_idx,:,:] = model_fits['actions']
        modout[model_name]['fit']['outcomes'][subj_idx,:,:] = model_fits['outcomes']
        modout[model_name]['fit']['rpes'][subj_idx,:,:] = model_fits['rpes']
        modout[model_name]['fit']['craving_ratings'][subj_idx,:,:] = model_fits['craving_ratings']
        modout[model_name]['fit']['pred_cravings'][subj_idx,:,:] = model_fits['pred_cravings']
        modout[model_name]['fit']['negll'][subj_idx] = model_fits['negll']

    # # Save output
    # if save_fit_path is not None:
    #     with open(f'{save_fit_path}/EMfit_{model_name}.pkl', 'wb') as f:
    #         pickle.dump(modout, f)
    
    return modout