# Description: Helper functions for the redux_sud_group_v2 scripts

# Importing libraries
from sys import path
import os
import time
import pickle
import datetime
from IPython.display import display

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler, normalize

import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
sns.set_style("white")
sns.set_context('poster')
sns.set_style({'font.family': 'Cambria Math'})
plt.rcParams['svg.fonttype'] = 'none'
import bambi as bmb

mm = 1/25.4*4

import warnings
warnings.filterwarnings('ignore')


### DEFINE PATHS
root_dir = "../"
data_dir = f'{root_dir}/demo_data/'
model_functions_path = f'{root_dir}/'
decision_results_dir = f'{root_dir}/outputs/decision/'
craving_results_dir = f'{root_dir}/outputs/craving/'
figure_dir = f'{root_dir}/outputs/figures/'
results_dir = f'{root_dir}/outputs/'
## Add model_functions to system path
path.append(model_functions_path)

### PLOTTING KWS
myboxplotprops = {
    'medianprops': dict(linewidth=2),
    'whiskerprops': dict(linewidth=2),
    'capprops': dict(linewidth=0),
    'boxprops': dict(fill=0, linewidth=2),
    'fliersize': 0,
    'gap': 0.35,
    'legend': False
}
mystripplotprops = {
    'alpha': 0.5,
    'dodge': True,
    'legend': False,
    'size': 5,
    'linewidth': 0.5
}

### CREATE REDCAP DATAFRAME
def create_master_demo_df():
    if os.path.exists(f'{data_dir}/redcap/master_demo_df.csv'):
        print('Loading master_demo_df...')
        return pd.read_csv(f'{data_dir}/redcap/master_demo_df.csv', index_col=0)
    
    print('Creating master_demo_df...')
    df_summary = pd.read_csv(f'{data_dir}/clean_all_group_df_summary.csv', index_col=0)
    longform = pd.read_csv(f'{data_dir}/clean_all_group_longform.csv', index_col=0)
    # Pretask loading for round 1 and 2
    path_to_pretask_redcap = f'{data_dir}/redcap/screening_surveys_r1r2.csv'
    pretask_df = pd.read_csv(path_to_pretask_redcap).filter(regex='assist3|edeq|locesb|socialmediause|prolific_pid$')
    # Posttask loading (posttask only for demographics)
    path_to_posttask_redcap = f'{data_dir}/redcap/posttask-all-drugs.csv'
    posttask_df = pd.read_csv(path_to_posttask_redcap)
    # Posttask Round 2 loading
    round2_path_to_posttask_redcap = f'{data_dir}/redcap/round2_posttask-all-drugs.csv'
    round2_posttask_df = pd.read_csv(round2_path_to_posttask_redcap)
    posttask_df = pd.concat([posttask_df, round2_posttask_df]).reset_index(drop=True)
    # Merge posttask and pretask
    master_demo_df = pd.DataFrame()
    tmp_df = pd.DataFrame()

    for i, pid in enumerate(df_summary[df_summary['Group']=='alcohol'].PID.unique()):
        retrieved = pretask_df[pretask_df['prolific_pid']==pid].copy()
        try:
            retrieved['audit_score'] = posttask_df[posttask_df['prolific_pid']==pid]['audit_score'].values[0]
            retrieved['age'] = posttask_df[posttask_df['prolific_pid']==pid]['age'].values[0]
            retrieved['sex'] = posttask_df[posttask_df['prolific_pid']==pid]['sex'].values[0]
            retrieved['edu_level'] = posttask_df[posttask_df['prolific_pid']==pid]['edu_level'].values[0]
            retrieved['income'] = posttask_df[posttask_df['prolific_pid']==pid]['income'].values[0]
            retrieved['ladder'] = posttask_df[posttask_df['prolific_pid']==pid]['ladder_us'].values[0]
            if posttask_df[posttask_df['prolific_pid']==pid]['race___1'].values[0]==1:
                retrieved['race'] = 1
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___2'].values[0]==1:
                retrieved['race'] = 2
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___3'].values[0]==1:
                retrieved['race'] = 3
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___4'].values[0]==1:
                retrieved['race'] = 4
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___5'].values[0]==1:
                retrieved['race'] = 5
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___6'].values[0]==1:
                retrieved['race'] = 6
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___7'].values[0]==1:
                retrieved['race'] = 7
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___8'].values[0]==1:
                retrieved['race'] = 8
            tmp_df = pd.concat([tmp_df, retrieved])
        except IndexError:
            continue

    tmp_df['ASSIST_Alcohol'] = tmp_df.filter(regex='assist3_q.*b').sum(axis=1, numeric_only=True, skipna=True)
    tmp_df['ASSIST_Freq'] = tmp_df['assist3_q2b_sc']
    tmp_df['ASSIST_Craving'] = tmp_df['assist3_q3b_sc']
    tmp_df['ASSIST_StopAttempt'] = tmp_df['assist3_q7b_sc']

    tmp_df = tmp_df[['prolific_pid', 'ASSIST_Alcohol', 'ASSIST_Freq', 'ASSIST_Craving', 'ASSIST_StopAttempt', 'age', 'sex', 'edu_level', 'income', 'race']]
    tmp_df = pd.merge(
        tmp_df,
        df_summary[
            (df_summary['Group']=='alcohol')
        ],
        left_on='prolific_pid',
        right_on='PID',
        how='inner'
    ).drop_duplicates(subset='prolific_pid').drop(columns=['prolific_pid'])
    master_demo_df = pd.concat([master_demo_df, tmp_df])

    tmp_df = pd.DataFrame()

    for i, pid in enumerate(df_summary[df_summary['Group']=='cannabis'].PID.unique()):
        retrieved = pretask_df[pretask_df['prolific_pid']==pid].copy()
        try:
            retrieved['audit_score'] = posttask_df[posttask_df['prolific_pid']==pid]['audit_score'].values[0]
            retrieved['age'] = posttask_df[posttask_df['prolific_pid']==pid]['age'].values[0]
            retrieved['sex'] = posttask_df[posttask_df['prolific_pid']==pid]['sex'].values[0]
            retrieved['edu_level'] = posttask_df[posttask_df['prolific_pid']==pid]['edu_level'].values[0]
            retrieved['income'] = posttask_df[posttask_df['prolific_pid']==pid]['income'].values[0]
            retrieved['ladder'] = posttask_df[posttask_df['prolific_pid']==pid]['ladder_us'].values[0]
            if posttask_df[posttask_df['prolific_pid']==pid]['race___1'].values[0]==1:
                retrieved['race'] = 1
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___2'].values[0]==1:
                retrieved['race'] = 2
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___3'].values[0]==1:
                retrieved['race'] = 3
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___4'].values[0]==1:
                retrieved['race'] = 4
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___5'].values[0]==1:
                retrieved['race'] = 5
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___6'].values[0]==1:
                retrieved['race'] = 6
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___7'].values[0]==1:
                retrieved['race'] = 7
            elif posttask_df[posttask_df['prolific_pid']==pid]['race___8'].values[0]==1:
                retrieved['race'] = 8
            tmp_df = pd.concat([tmp_df, retrieved])
        except IndexError:
            continue

    tmp_df['ASSIST_Cannabis'] = tmp_df.filter(regex='assist3_q[0-9]*c_sc').sum(axis=1, numeric_only=True, skipna=True)
    tmp_df['ASSIST_Freq'] = tmp_df['assist3_q2c_sc']
    tmp_df['ASSIST_Craving'] = tmp_df['assist3_q3c_sc']
    tmp_df['ASSIST_StopAttempt'] = tmp_df['assist3_q7c_sc']

    tmp_df['CAST_Cannabis'] = tmp_df.filter(regex='cast').sum(axis=1, numeric_only=True, skipna=True) - 6
    tmp_df['SDS_Cannabis'] = tmp_df.filter(regex='sds').sum(axis=1, numeric_only=True, skipna=True) - 5

    tmp_df = tmp_df[['prolific_pid', 'ASSIST_Cannabis', 'ASSIST_Freq', 'ASSIST_Craving', 'ASSIST_StopAttempt', 'age', 'sex', 'edu_level', 'income', 'race']]
    tmp_df = pd.merge(
        tmp_df,
        df_summary[
            (df_summary['Group']=='cannabis')
        ],
        left_on='prolific_pid',
        right_on='PID',
        how='inner'
    ).drop_duplicates(subset='prolific_pid').drop(columns=['prolific_pid'])
    master_demo_df = pd.concat([master_demo_df, tmp_df])

    master_demo_df['DEMO_age'] = master_demo_df['age']

    def sex_filter(arr):
        if arr.sex==1:
            return 'Male'
        elif arr.sex==2:
            return 'Female'
        elif arr.sex==3:
            return 'Other'
    master_demo_df['DEMO_sex'] = master_demo_df.filter(regex='^sex').apply(sex_filter, axis=1)

    def edu_filter(arr):
        if arr.edu_level==1:
            return '< High School'
        elif arr.edu_level==2:
            return 'Some HS'
        elif arr.edu_level==3:
            return 'High School'
        elif arr.edu_level==4:
            return 'Some college'
        elif arr.edu_level==5:
            return 'College'
        elif arr.edu_level==6:
            return 'Graduate'

    master_demo_df['DEMO_edu'] = master_demo_df.filter(regex='^edu_level').apply(edu_filter, axis=1)

    def income_filter(arr):
        if arr.income==1:
            return '<10k'
        elif arr.income==2:
            return '10-20k'
        elif arr.income==3:
            return '20-30k'
        elif arr.income==4:
            return '30-40k'
        elif arr.income==5:
            return '40-50k'
        elif arr.income==6:
            return '50-60k'
        elif arr.income==7:
            return '60-70k'
        elif arr.income==8:
            return '70-80k'
        elif arr.income==9:
            return '80-90k'
        elif arr.income==10:
            return '90-100k'
        elif arr.income==11:
            return '100-150k'
        elif arr.income==12:
            return '>150k'

    master_demo_df['DEMO_income'] = master_demo_df.filter(regex='^income').apply(income_filter, axis=1)

    def race_filter(arr):
        if arr.race==1:
            return 'American Indian or Alaska Native'
        elif arr.race==2:
            return 'Asian'
        elif arr.race==3:
            return 'Black or African American'
        elif arr.race==4:
            return 'Hispanic or Latino'
        elif arr.race==5:
            return 'Multiracial'
        elif arr.race==6:
            return 'Native Hawaiian or Other Pacific Islander'
        elif arr.race==7:
            return 'White'
        elif arr.race==8:
            return 'Other'

    master_demo_df['DEMO_race'] = master_demo_df.filter(regex='^race').apply(race_filter, axis=1)

    master_demo_df['Group'] = master_demo_df['Group'].str.capitalize()

    if not os.path.exists(f'{data_dir}/redcap/master_demo_df.csv'):
        master_demo_df.to_csv(f'{data_dir}/redcap/master_demo_df.csv')

    return master_demo_df

### PLOTTING FUNCTIONS

def plot_posterior_predictive_redux(bmb_model_dict, performance_df, reg_params, group_name, figsize):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if group_name == 'alcohol':
        set2_num = 2
    elif group_name == 'cannabis':
        set2_num = 5
    best_model_name = str(performance_df['Model'].to_numpy()[0])
    for i, prefix in enumerate(['demo', 'comp', 'magnos', 'democomp', 'demomagnos']):
        if best_model_name.startswith(prefix):
            c = sns.color_palette('Set2')[set2_num]
            alph = 1
            lw = 5
        else:
            c = 'grey'
            alph = 0.2
            lw = 3
        best_representative_model = performance_df[performance_df['Model'].str.startswith(prefix)].iloc[0]['Model']
        best_model_results = bmb_model_dict[best_representative_model]
        
        sns.regplot(
            x=best_model_results.observed_data[reg_params["dependent_var"]].values,
            y=np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            color=c, ci=95, ax=ax, scatter_kws=dict(alpha=alph, s=70), line_kws=dict(lw=lw, alpha=alph), label=prefix
            #marker='o'
        )
        # annotate with correlation and p-value
        r, p = stats.pearsonr(best_model_results.observed_data[reg_params["dependent_var"]].values, np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0))
        # ax.text(0.05, 0.95-i*0.05, f'{prefix.capitalize()}: r={r:.3f}, p={p:.3f}', transform=ax.transAxes, va='top', ha='left', fontsize=18)
        print(f'{prefix.capitalize()}: r={r:.3f}, p={p:.3f}')
    
    sns.despine()
    # ax.set_xlabel(f'True Score')
    # ax.set_ylabel(f'Predicted Score')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.text(0.05, 0.95, f'{group_name.capitalize()}', transform=ax.transAxes, va='top', ha='left', fontsize=18)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    return fig, ax

def performance_comp_redux(sm_model_dict, bmb_model_dict, figsize, bambi_flag=False):
    performance_df = pd.DataFrame(columns=['Model', 'R2', 'R2_adj', 'AIC', 'BIC'])
    for model in sm_model_dict:
        sm_results = sm_model_dict[model]
        performance_df = pd.concat([
            performance_df,
            pd.DataFrame({
                'Model': [model],
                'R2': [sm_results.rsquared],
                'R2_adj': [sm_results.rsquared_adj],
                'AIC': [sm_results.aic],
                'BIC': [sm_results.bic]
            })
        ], axis=0)
    performance_df = performance_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    display(performance_df)
    
    if bambi_flag:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        name_dict = {
            'demo': 'Demo-R',
            'comp': 'Comp-R',
            'magnos': 'Agnostic-R',
            'democomp': 'CompDemo-R',
            'demomagnos': 'AgnosticDemo-R'
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_compare = az.compare(bmb_model_dict, ic='waic')
            display(df_compare)
            az.plot_compare(df_compare, insample_dev=True, plot_ic_diff=True, ax=ax, legend=False);
            ax.set_title(f'')
            # ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.set_xticklabels([])
            ax.set_yticklabels([])
    return performance_df, fig, ax
