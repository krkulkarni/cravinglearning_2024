import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from IPython.display import display
import arviz as az
import bambi as bmb
import statsmodels.formula.api as smf

def kk_boxplot(df, xvar, yvar, huevar, figsize=(16, 6), num_x=None, **kwargs):
    
    if num_x is None:
        num_x = len(df[xvar].unique())
    fig, ax = plt.subplots(1, num_x, figsize=figsize)

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

    for i, xval in enumerate(df[xvar].unique()):
        xvar_filtered_df = df[df[xvar] == xval]
        sns.stripplot(
            xvar_filtered_df,
            x=xvar, y=yvar, hue=huevar,
            ax=ax[i],
            palette=[sns.palettes.color_palette('Set2')[2], sns.palettes.color_palette('Set2')[5]],
            **mystripplotprops
        )
        sns.boxplot(
            xvar_filtered_df,
            x=xvar, y=yvar, hue=huevar,
            ax=ax[i],
            palette=[sns.palettes.color_palette('Set2')[2], sns.palettes.color_palette('Set2')[5]],
            **myboxplotprops
        )
        # sns.boxplot(
        #     xvar_filtered_df,
        #     x=xvar, y=yvar, hue=huevar,
        #     ax=ax[i],
        #     # palette=[sns.palettes.color_palette('Set2')[2], sns.palettes.color_palette('Set2')[5]],
        #     color='black',
        #     showfliers=False, whis=0.5,
        #     boxprops=dict(fill=None),
        #     gap=0.5,
        #     **kwargs
        # )
        ax[i].set_title(f'{xvar} = {xval}')
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        # ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)
        # ax[i].get_legend().remove()
        # set ylim to min and max of xvar data
        ax[i].set_ylim(xvar_filtered_df[yvar].min()-0.1*(xvar_filtered_df[yvar].max()-xvar_filtered_df[yvar].min()), xvar_filtered_df[yvar].max()+0.3*(xvar_filtered_df[yvar].max()-xvar_filtered_df[yvar].min()))

        plt.setp(ax[i].spines.values(), linewidth=1.5)
    
    # fig.suptitle('Decision parameter estimates by group')
    # sns.despine()
    # plt.tight_layout()
    return fig, ax

## Plotting functions
def performance_comp(sm_model_dict, bmb_model_dict, figsize, bambi_flag=False):
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
            # 'agnostic': 'Agnostic-R',
            # 'comp': 'Comp-R',
            # 'combined': 'Joint-R'
            'demo': 'Demo-R',
            'comp': 'Comp-R',
            'magnos': 'Agnostic-R',
            'democomp': 'CompDemo-R',
            'demomagnos': 'AgnosticDemo-R'
        }
        filtered_bmb_dict = {k: v for k, v in bmb_model_dict.items() if 'filtered' in k}
        model_agnostic_present = False
        # for k, v in filtered_bmb_dict.items():
        #     if 'agnostic' in k:
        #         model_agnostic_present = True
        # if not model_agnostic_present:
        #     filtered_bmb_dict['model_agnostic_filtered'] = bmb_model_dict['model_agnostic']
        # filtered_bmb_dict_renamed = {}
        # for k, v in filtered_bmb_dict.items():
        #     if 'agnostic' in k:
        #         filtered_bmb_dict_renamed[f'{name_dict["agnostic"]}'] = v
        #     elif 'combined' in k:
        #         filtered_bmb_dict_renamed[f'{name_dict["combined"]}'] = v
        #     else:
        #         filtered_bmb_dict_renamed[f'{name_dict["comp"]}'] = v
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # df_compare = az.compare(filtered_bmb_dict_renamed, ic='waic')
            df_compare = az.compare(bmb_model_dict, ic='waic')
            display(df_compare)
            az.plot_compare(df_compare, insample_dev=True, plot_ic_diff=True, ax=ax, legend=False);
            ax.set_title(f'')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    return performance_df, fig, ax

def plot_r2_adj(performance_df):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(x='Model', y='R2_adj', data=performance_df, dodge=False, ax=ax, palette='Set2', edgecolor='black', linewidth=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12);

def plot_sig_predictive(sm_model_dict, bmb_model_dict, reg_params):
    cand_param_list = reg_params['full_param_list'] + reg_params['full_nonmodel_param_list'] + reg_params['full_demo_list']
    fig, ax = plt.subplots(figsize=(17, 5), ncols=3, nrows=1)
    i_count = -1
    for i, model_name in enumerate(bmb_model_dict.keys()):
        if 'filtered' not in model_name:
            continue
        if i>8:
            break
        else:
            i_count += 1
            if 'model_agnostic' in model_name:
                ax[i_count].set_title('model_agnostic')
            elif 'combined' in model_name:
                ax[i_count].set_title('Combined')
            else:
                ax[i_count].set_title('Model parameters')
        vars = []
        summary_model_df = az.summary(bmb_model_dict[model_name])
        candidate_vars = np.array([
            elem for elem in az.summary(bmb_model_dict[model_name]).index if elem in cand_param_list
        ])
        for var in candidate_vars:
            if np.abs(summary_model_df.loc[var]['hdi_3%'] + summary_model_df.loc[var]['hdi_97%']) > np.abs(summary_model_df.loc[var]['hdi_97%']):
                vars.append(var)
        az.plot_forest(
            bmb_model_dict[model_name], 
            var_names=vars,
            figsize=(5,4), combined=True, 
            kind='ridgeplot', ridgeplot_alpha=0.3, 
            hdi_prob=0.89,
            # ridgeplot_quantiles=[.25, .5, .75],
            ridgeplot_overlap=1, 
            ridgeplot_truncate=False,
            colors='gray',
            ax=ax[i_count],
        )
        # plt.xlim(-15, 15)
        # ax[i_count].set_title(model_name)
        ax[i_count].axvline(0, color='k', linestyle='--')
        ax[i_count].set_yticklabels(ax[i_count].get_yticklabels(), rotation=45, ha='right')
    plt.tight_layout()

def plot_posterior_predictive(bmb_model_dict, performance_df, reg_params, group_name, figsize):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if group_name == 'alcohol':
        set2_num = 2
    elif group_name == 'cannabis':
        set2_num = 5
    for i, prefix in enumerate(['demo', 'comp', 'magnos', 'democomp', 'demomagnos']):
        if prefix=='demo':
            c = 'grey'
            alph = 0.3
        elif prefix=='comp':
            c = sns.color_palette('Set2')[set2_num]
            alph = 1
        elif prefix=='magnos':
            c = 'grey'
            alph = 0.3
        elif prefix=='democomp':
            c = 'grey'
            alph = 0.3
        elif prefix=='demomagnos':
            c = 'grey'
            alph = 0.3
        best_representative_model = performance_df[performance_df['Model'].str.startswith(prefix)].iloc[0]['Model']
        best_model_results = bmb_model_dict[best_representative_model]
        print(best_representative_model)

        # az.plot_hdi(
        #     best_model_results.observed_data[reg_params["dependent_var"]].values,
        #     best_model_results.posterior_predictive[reg_params["dependent_var"]].values,
        #     hdi_prob=0.84, smooth=True, color=c, ax=ax,
        #     fill_kwargs={'alpha': 0.03}
        # )
        # az.plot_hdi(
        #     best_model_results.observed_data[reg_params["dependent_var"]].values,
        #     best_model_results.posterior_predictive[reg_params["dependent_var"]].values,
        #     hdi_prob=0.68, smooth=True, color=c, ax=ax,
        #     fill_kwargs={'alpha': 0.1}
        # )
        sns.regplot(
            x=best_model_results.observed_data[reg_params["dependent_var"]].values,
            y=np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            color=c, ci=95, ax=ax, scatter_kws=dict(alpha=alph), line_kws=dict(lw=2, alpha=alph), label=prefix
            #marker='o'
        )
    
    sns.despine()
    # ax.set_xlabel(f'True Score')
    # ax.set_ylabel(f'Predicted Score')
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.text(0.05, 0.95, f'{group_name.capitalize()}', transform=ax.transAxes, va='top', ha='left', fontsize=18)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig, ax

def plot_sig_improvement(bmb_model_dict, reg_params, perf_df):
    best_model_name = perf_df.iloc[0]['Model']
    best_model_results = bmb_model_dict[best_model_name]
    try:
        predicted = np.hstack((
            np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            np.vstack(bmb_model_dict['model_agnostic_filtered'].posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0)
        ))
    except KeyError:
        predicted = np.hstack((
            np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            np.vstack(bmb_model_dict['model_agnostic'].posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0)
        ))
    true = np.hstack((
        best_model_results.observed_data[reg_params["dependent_var"]].values,
        best_model_results.observed_data[reg_params["dependent_var"]].values
    ))
    model_labels = np.hstack((
        # np.repeat(r2_df.loc[0]['Model'], len(sample_model_df)),
        np.repeat(1, len(best_model_results.observed_data[reg_params["dependent_var"]].values)),
        # np.repeat('model-free', len(sample_model_df))
        np.repeat(0, len(best_model_results.observed_data[reg_params["dependent_var"]].values))
    ))
    predict_df = pd.DataFrame({
        'predicted': predicted,
        'true': true,
        'model': model_labels
    })

    equation = 'predicted ~ true * model'
    print(equation)
    pred_model = bmb.Model(equation, predict_df)
    results = pred_model.fit()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    az.plot_forest(
        results, 
        filter_vars='regex',
        var_names=['^true'],
        figsize=(5,2), combined=True, 
        kind='ridgeplot', ridgeplot_alpha=0.3, 
        hdi_prob=0.89,
        # ridgeplot_quantiles=[.25, .5, .75],
        ridgeplot_overlap=1, 
        ridgeplot_truncate=False,
        colors='gray',
        ax=ax,
    )
    ax.axvline(0, color='k', linestyle='--')
    plt.tight_layout()
    display(smf.ols(equation, predict_df).fit().summary())

def plot_sig_improvement2(bmb_model_dict, reg_params, perf_df):
    best_model_name = perf_df.iloc[0]['Model']
    best_model_results = bmb_model_dict[best_model_name]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), sharey=True)
    try:
        sns.regplot(
            x=best_model_results.observed_data[reg_params["dependent_var"]].values,
            y=np.vstack(bmb_model_dict['model_agnostic_filtered'].posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            color='gray', ci=89, ax=ax, scatter_kws=dict(s=50, alpha=0.3), line_kws=dict(lw=2, alpha=0.3)
        )
    except KeyError:
        sns.regplot(
            x=best_model_results.observed_data[reg_params["dependent_var"]].values,
            y=np.vstack(bmb_model_dict['model_agnostic'].posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
            color='gray', ci=89, ax=ax, scatter_kws=dict(s=50, alpha=0.3), line_kws=dict(lw=2, alpha=0.3)
        )
    sns.regplot(
        x=best_model_results.observed_data[reg_params["dependent_var"]].values,
        y=np.vstack(best_model_results.posterior_predictive[reg_params["dependent_var"]].values).mean(axis=0),
        color=sns.palettes.color_palette('Set2')[1], ci=89, ax=ax, scatter_kws=dict(s=50)
    )
    sns.despine()
    # ax.set_xlabel(f'True {reg_params["dependent_var"].capitalize()}')
    # ax.set_ylabel(f'Predicted {reg_params["dependent_var"].capitalize()}')
    # ax.set_xlabel(f'True score')
    # ax.set_ylabel(f'Predicted score')
    # ax.text(0.05, 1.01, f'Best model', transform=ax.transAxes, va='top', ha='left', bbox=dict(edgecolor=sns.palettes.color_palette('Set2')[1], facecolor='white'))
    # ax.text(0.05, 0.84, 'Model-free', transform=ax.transAxes, va='top', ha='left', bbox=dict(edgecolor='gray', facecolor='white'))
    plt.tight_layout()

# Function to generate the dataframe of parameter estimates and confidence intervals from statsmodels model
def generate_param_estimates_df(sm_results, model_name):
    x = pd.read_html(sm_results.summary().tables[1].as_html())[0]
    y = x.iloc[0].astype(str)
    y[0] = 'Predictor'
    x.columns = y
    x = x.iloc[1:]
    x = pd.concat([
        x['Predictor'],
        x[[elem for elem in x.columns if 'Predictor' not in elem]].astype(str).astype(float)
    ], axis=1)
    x = x[x['Predictor']!='Intercept']
    x = x.sort_values(by='P>|t|', ascending=False)
    x['Model'] = model_name
    x = x[['Model', 'Predictor', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']]
    x.columns = ['Model', 'Predictor', 'coef', 'std err', 't', 'P>|t|', 'lower', 'upper']
    x['mean'] = x['coef']
    x['ci'] = (x['upper'] - x['lower']) / 2
    x['pval'] = x['P>|t|']
    x = x[['Model', 'Predictor', 'mean', 'ci', 'pval']]
    return x.sort_values(by='pval', ascending=True).reset_index(drop=True)

# Function to plot from mean and confidence interval from a dataframe of multiple parameters
def plot_from_df(glm_param_df, ax, group):
    sns.scatterplot(
        x=glm_param_df['mean'],
        y=glm_param_df.index,
        ax=ax, color='grey'
    )
    ax.errorbar(
        x=glm_param_df['mean'],
        y=glm_param_df.index,
        xerr=glm_param_df['ci'],
        elinewidth=3, capsize=3, capthick=3, fmt='none',
        ecolor='grey'
    )
    sig_df = glm_param_df[glm_param_df['pval']<0.05]
    # display(sig_df)
    sns.scatterplot(
        x=sig_df['mean'],
        y=sig_df.index,
        ax=ax, color='red'
    )
    ax.errorbar(
        x=sig_df['mean'],
        y=sig_df.index,
        xerr=sig_df['ci'],
        elinewidth=3, capsize=3, capthick=3, fmt='none',
        ecolor='red'
    )
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter name')
    ax.set_title(f'Parameter values for {group} group', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    # Add y-axis labels with parameter names
    ax.set_yticklabels(glm_param_df['Predictor'], rotation=45)
    ax.set_yticks(glm_param_df.index)
    sns.despine()