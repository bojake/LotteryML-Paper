#
# jackpotanalysis.py - (c) Beyond Ordinary Software Solutions, all rights reserved. This file
# is part of the LotteryML paper repository and is subject to the GNU General Public License v3.0.
# 
import os
import math
import numpy as np
import pandas as pd
import argparse
import LottoConfig as lc
import datetime as dt
import lottostats as ls
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PathCollection
import matplotlib as mpl
from scipy.stats import ks_2samp, chisquare
from scipy.stats import chi2_contingency
from scipy.stats import wasserstein_distance
from scipy.special import comb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
import PowerballEra as pb
import pickle as pkl
from scipy.stats import beta, kstest
import statsmodels.api as sm
import montecarlo as mc
from LottoConfig import LottoConfigEntry
from scipy import stats
import matplotlib.dates as mdates


def split_into_eras(df_jackpot : pd.DataFrame, lotto : str) -> list:
    '''
    Splits the jackpot data into eras based on the LottoConfig time splits. Each split is
    saved as a csv file.
    :param df_jackpot: The DataFrame containing the jackpot data
    :param lotto: The lottery name, e.g. 'PowerBall
    :return: A list of PowerballEra objects for each era
    '''
    eras = []
    time_splits = lc.LottoConfig().lotto_splits(lotto)
    if time_splits is not None :
        eras = []
        for config in time_splits:
            start_dt = config.start_date
            end_dt = config.end_date
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if end_dt is None:
                print(f"Time split for {lotto} from {start_dt} to present")
                df_split = df_jackpot[df_jackpot['Date'] >= start_dt]
            else:
                print(f"Time split for {lotto} from {start_dt} to {end_dt}")
                df_split = df_jackpot[(df_jackpot['Date'] >= start_dt) & (df_jackpot['Date'] < end_dt)]
            df_split = df_split[df_split.D1.notnull()]
            print(f"Analyzing {df_split.shape[0]} draws in this era")
            if end_dt is None :
                end_dt = dt.datetime.today()
            df_split.to_csv(os.path.join(lotto, f"{lotto}-era-split-{start_dt:%Y%m%d}-{end_dt:%Y%m%d}.csv"), index=False)
            era = pb.PowerballEra('PowerBall', config)
            era.analyze(df_split)
            eras.append(era)
            csv_filename = era.save_filename('details')
            era.details.to_csv(os.path.join(lotto, csv_filename),index=False)
            csv_filename = era.save_filename('probs')
            era.probs.to_csv(os.path.join(lotto, csv_filename),index=False)
    return eras

def run_analysis(filename : str, lotto : str, mc_runs : int = 0) -> None :
    '''
    Splits the data into eras. Runs the monte carlo simulation if requested and then created the 
    beta graphs for each era.
    :param filename: The source data file containing the lottery ticket history
    :param lotto: The lottery name, e.g. 'PowerBall'
    :param mc_runs: The number of Monte Carlo runs to perform for each era, default is 0 (no MC runs)
    '''
    df_jackpot = None
    if os.path.exists(filename) :
        df_jackpot = pd.read_csv(filename)
    else:
        print(f'No jackpot analysis file {filename} found, skipping jackpot analysis.')
        return
    print(df_jackpot.describe())
    os.makedirs(lotto, exist_ok=True)
    n_jackpot_tickets = min(3074, df_jackpot.shape[0] if df_jackpot is not None else 3074) - 1
    print(f'Running jackpot analysis for {lotto} with {n_jackpot_tickets} tickets')
    df_jackpot['Date'] = pd.to_datetime(df_jackpot['Date'], format='%Y-%m-%d')
    eras = split_into_eras(df_jackpot, lotto)
    #
    # If we are running the monte carlo simulation, then invoke that function on the era
    # for each era that was created in the split.
    #
    if mc_runs > 0 :
        for era in eras :
            era.run_montecarlo(runs=mc_runs)
    with open(os.path.join(lotto, 'jackpot_analysis.manifest'), 'w') as fp:
        for era in eras :
            print("=================================================================================")
            print(f"Era: {era.era_string}, analyzing {era.T} draws")
            era.report()
            path_to_save = os.path.join(lotto, era.save_filename("objects").replace('.csv', '.pkl'))
            with open(path_to_save, 'wb') as era_fp:
                pkl.dump(era,era_fp)
            fp.write(f"{path_to_save}\n")
    for era in eras :
        beta_graph(era)

def qq_annotated(M:int, N:int, k:int, draws:pd.DataFrame, lotto:str, alpha:float=0.05,axes:plt.Axes=None) -> None:
    """
    Perform KS test and generate an annotated Q-Q plot comparing empirical draws
    to the theoretical Beta(k, N+1-k) distribution. Saves PNG with:
      - dashed 45° line
      - confidence band
      - annotation of max deviation (KS D)
      - inset zooms on both tails
    """
    T = draws.shape[0]
    # 1. KS test
    a, b = k, N + 1 - k
    # transform draws into [0,1] via CDF of Beta
    u = beta.cdf((draws - 1) / (M - 1), a, b)
    D, p_value = kstest(u, 'uniform')
    KS_power = ls.ks_power(T=T,alpha=alpha,delta=D)


    # 2. Prepare Q-Q data (normalized scale 0..1)
    n = len(u)
    # theoretical quantiles at midpoints
    qs = (np.arange(n) + 0.5) / n
    th_q = beta.ppf(qs, a, b)
    emp_q = np.sort((draws - 1) / (M - 1))

    # 3. Plot
    fig = None
    if axes is None :
      fig, ax = plt.subplots(figsize=(6, 6))
    else :
       ax = axes
    # confidence band (KS critical)
    ks_crit = 1.36 / np.sqrt(n)  # approx for alpha=0.05
    x_vals = np.array([0, 1])
    ax.fill_between(x_vals, x_vals - ks_crit, x_vals + ks_crit,
                    color='lightgray', alpha=0.5, label=f'{alpha*100:.0f}% conf. band')
    # 45° reference line
    ax.plot(x_vals, x_vals, 'r--', linewidth=1, label='y = x')

    # Q-Q scatter
    ax.scatter(th_q, emp_q, s=20, color='C0', alpha=0.7, label='Quantiles')

    # annotate max deviation
    idx = np.argmax(np.abs(emp_q - th_q))
    x0, y0 = th_q[idx], emp_q[idx]
    # arrow from line to point
    ax.annotate('', xy=(x0, y0), xytext=(x0, x0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(0.5, 0.90,
            f"KS D = {D:.3f} at {KS_power*100.0:2.2f}%\np = {p_value:.3f}\n$\\alpha$ = {alpha:.2f}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_title(f"Annotated Q-Q Plot: Position D{k}")
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel(f'{lotto} Quantiles')

    # 4. Tail zoom insets
    # lower tail zoom
    axins_low = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=2)
    mask_low = qs <= 0.1
    axins_low.scatter(th_q[mask_low], emp_q[mask_low], s=20, color='C0', alpha=0.7)
    axins_low.plot(x_vals, x_vals, 'r--', linewidth=1)
    axins_low.set_xlim(0, 0.1)
    axins_low.set_ylim(0, 0.1)
    axins_low.set_xticks([])
    axins_low.set_yticks([])
    axins_low.set_title('Lower 10% tail', fontsize=8)

    # upper tail zoom
    axins_high = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=2)
    mask_high = qs >= 0.9
    axins_high.scatter(th_q[mask_high], emp_q[mask_high], s=20, color='C0', alpha=0.7)
    axins_high.plot(x_vals, x_vals, 'r--', linewidth=1)
    axins_high.set_xlim(0.9, 1)
    axins_high.set_ylim(0.9, 1)
    axins_high.set_xticks([])
    axins_high.set_yticks([])
    axins_high.set_title('Upper 10% tail', fontsize=8)

    # legend
    ax.legend(loc='lower center')


def qq_era_graph(era : pb.PowerballEra) -> None:
    '''
    Generatel Q-Q graphs versus the order statistics of the beta distribution
    :param era: The PowerballEra object containing the draw data
    '''
    qq_fig, qq_axes = plt.subplots(3, 2, figsize=(15, 15))
    qq_fig.suptitle(
        f"{era.title} Draw Positions Q-Q: Empirical vs. Theoretical $\\beta$ Order-Statistics\n{era.era_string}",
        fontsize=14, y=1.02
    )

    for pos in range(5):
        col   = era.config.cols[pos]
        M     = era.config.maxval[pos]
        N     = 5
        k     = pos + 1
        draws = era.data[col]
        ax = qq_axes[pos//2,pos%2]
        qq_annotated(M, N, k, draws.values,lotto=era.lotto, axes=ax)

    qq_fig.delaxes(qq_axes[2,1])
    qq_fig.tight_layout()
    fig_filename = f"{era.lotto}-QQ-PMF-{era.era_string}.png"
    qq_fig.savefig(os.path.join(era.lotto,fig_filename), facecolor=qq_fig.get_facecolor(),bbox_inches='tight')
    plt.close(qq_fig)


def pmf_graph(era : pb.PowerballEra) -> None :
    '''
    Generate the PMF graph for each draw position, showing the empirical PMF and the error, 
    with 5 panels in the graph in a 3x2 grid.
    :param era: The PowerballEra object containing the draw data
    '''
    lotto_config = era.config
    p,x = era.pmf # ls.compute_pmf(df_data=era.data, lotto_config=era.config)

    error_mean = x[:,4].mean()
    error = x[:,4]
    epfig, epaxes = plt.subplots(3, 2, figsize=(15, 10))
    epfig.suptitle(
        f"{era.title} Empirical PMF {era.era_string} $\\bar\\epsilon$={error_mean*100:.2f}% at 95%",
        fontsize=14, y=1.02
    )
    mustd = []
    for pos in range(0,len(lotto_config.cols)) :
        col = lotto_config.cols[pos]
        n = lotto_config.maxval[pos]
        axes = epaxes[pos//2,pos%2]
        err = x[pos,4]
        axes.plot(p[:n,pos], label=f'{col} $\\epsilon$={err*100:.2f}% at 95%')
        axes.set_xlabel("Position Value")
        axes.set_ylabel("Probability")
        axes.set_title(f"PMF of {era.lotto} Position {col}")
        mean_x = x[pos,0]
        std_x = x[pos,1]
        mustd.append((mean_x,std_x))
        axes.axvline(x=mean_x,color='r')
        axes.axvline(x=mean_x+std_x,color='r',linestyle=':')
        if mean_x - std_x > 1 :
            axes.axvline(x=mean_x-std_x,color='r',linestyle=':')
        seq = np.arange(1,n+1)
        axes.set_xticks(seq,minor=True)
        axes.legend()

    epfig.tight_layout()
    fig_filename = f"{era.lotto}-PMF-{era.era_string}.png"
    epfig.savefig(os.path.join(era.lotto,fig_filename), facecolor=epfig.get_facecolor(),bbox_inches='tight')

'''
Generate the A/B graph with a 95% CI band
'''
def beta_graph_with_ci(era, ref_probs = None, ref_label = None) -> None:
    lotto_config = era.config
    if ref_probs is None :
        ref_label = 'Beta(k,6-k)'
        ref_probs, _ = ls.compute_beta(df_data=era.details,lotto_config=era.config)
        rp = np.zeros((ref_probs.shape[0], len(lotto_config.cols)))
        rp[:,:ref_probs.shape[1]] = ref_probs
        rp[:,len(lotto_config.cols)-1] = np.ones_like(ref_probs[:,0])/ lotto_config.maxval[-1]
        ref_probs = rp

    epfig, epaxes = plt.subplots(6, 2, figsize=(15, 15),gridspec_kw={'height_ratios': [3,1,3,1,3,1]})
    epfig.suptitle(
        f"{era.title} Empirical PMF vs. {ref_label} With 95% CI in {era.era_string}",
        fontsize=14, y=1.02
    )
    emp_probs,xxx = era.pmf # ls.compute_pmf(df_data=era.data,lotto_config=era.config)
    metrics = ls.histogram_comparison(ref_probs, emp_probs, lotto_config)
    kl_pos = era.kl_by_position
    tv_pos = era.tv_by_position
    chi_p = era.chi_p_by_position
    cpos = 0
    for r in range(0,6,2) :
        for xp,axes in enumerate(epaxes[r]) :
            pos = cpos + xp
            col = lotto_config.cols[pos]
            n = lotto_config.maxval[pos]
            x_vals = np.arange(1, lotto_config.maxval[pos]+1)
            p0 = ref_probs[:n,pos]
            T = era.T
            moe = 1.96 * np.sqrt(p0 * (1 - p0) / T)
            m1, m2 = np.clip(p0 - moe,0,1), np.clip(p0 + moe,0,1)
            outliers = (emp_probs[:n,pos] < m1) | (emp_probs[:n,pos] > m2)
            # axes = epaxes[pos//2,pos%2]
            axes.plot(x_vals,ref_probs[:n,pos], color='g', linestyle='-', label=f'{ref_label} {lotto_config.cols[pos]}')
            axes.plot(x_vals,emp_probs[:n,pos], color='r', linestyle='--',label=f'Empirical {lotto_config.cols[pos]}')
            # From chat, the simulation envelope
            axes.fill_between(x_vals, m1, m2, color='C3', alpha=0.2, label='95% envelope')
            axes.set_ylabel("Probability")
            axes.scatter(x_vals[outliers], emp_probs[:n,pos][outliers], color='red', zorder=3, label='Outside 95% band')
            axes.set_title(f"PMF For Position {col}, KL={kl_pos[pos]:.3f}, TV={tv_pos[pos]:.3f}, $X^2 p$={chi_p[pos]:.3f}")
            mu = ref_probs[:,pos].mean()
            std = ref_probs[:,pos].std()
            mean_x,std_x,_,_,_ = era.pmf[1][pos]
            print(f'X: {col} mean={mean_x}, std={std_x}')
            axes.axvline(x=mean_x,color='r')
            axes.axvline(x=mean_x+std_x,color='r',linestyle=':')
            if mean_x - std_x > 1 :
                axes.axvline(x=mean_x-std_x,color='r',linestyle=':')
            wasser = metrics[col]['W']
            bhatta = metrics[col]['B']
            hellinger = metrics[col]['H']
            jensen = metrics[col]['JS']
            axes.set_xlabel(f"W={wasser:.3f} B={bhatta:.3f} H={hellinger:.3f} JS={jensen:.3f}")
            seq = np.arange(1,lotto_config.maxval[5]+1,3)
            axes.set_xticks(seq,minor=True)
            axes.legend()
        for xp,axes in enumerate(epaxes[r+1]) :
            pos = cpos + xp
            col = lotto_config.cols[pos]
            n = lotto_config.maxval[pos]
            x_vals = np.arange(1, lotto_config.maxval[pos]+1)
            dd = emp_probs[:,pos] - ref_probs[:,pos]
            label = f'Empirical - {ref_label}'
            if pos == len(lotto_config.cols)-1:
                label = 'Empirical - Uniform'
            axes.plot(x_vals,dd[:n], color='g', linestyle='-', label=f'Empirical - {label}')
            axes.axhline(y=0, color='r', linestyle='--')
            axes.set_title(f"Empirical - {label}")
            seq = np.arange(1,lotto_config.maxval[5]+1,3)
            axes.set_xticks(seq,minor=True)
        cpos += 2

    epfig.tight_layout()
    fig_filename = f"PMF-CI-{ref_label}-to-empirical-{era.era_string}.png"
    epfig.savefig(os.path.join(era.lotto,fig_filename), facecolor=epfig.get_facecolor(),bbox_inches='tight')
    print_positional_statistics(era)

def print_positional_statistics(era:pb.PowerballEra, ref_probs:np.ndarray = None) -> None:
    '''
    Print out a summary table that compares the empirical probabilities of the era against
    the beta distribution probabilities. The summary table includes KL divergence, Total Variation,
    Chi-squared p-value, Wasserstein distance, Bhattacharyya distance, Hellinger distance, and Jensen-Shannon divergence.
    :param era: The PowerballEra object containing the draw data
    :param ref_probs: The reference probabilities to compare against, if None, the beta distribution is used
    '''
    emp_probs,xxx = era.pmf
    lotto_config = era.config
    if ref_probs is None :
        ref_probs, _ = ls.compute_beta(df_data=era.details,lotto_config=era.config)
        rp = np.zeros((ref_probs.shape[0], len(lotto_config.cols)))
        rp[:,:ref_probs.shape[1]] = ref_probs
        rp[:,len(lotto_config.cols)-1] = np.ones_like(ref_probs[:,0])/ lotto_config.maxval[-1]
        ref_probs = rp
    metrics = ls.histogram_comparison(ref_probs, emp_probs, lotto_config)
    kl_pos = era.kl_by_position
    tv_pos = era.tv_by_position
    chi_p = era.chi_p_by_position
    ks_results = []
    D_max = -999
    P_max = -999
    for pos in range(len(lotto_config.cols)-1):
        col   = lotto_config.cols[pos]
        M     = lotto_config.maxval[pos]
        N     = 5
        k     = pos + 1
        draws = era.data[col]
        D, p  = ls.kolmogorov(M, N, k, draws.values)
        ks_results.append((k, D, p))
    #
    # Print the summary table too
    #
    print(f'+=========================================================================================+')
    print(f'| {era.era_string:>25}                                                               |')
    print(f'| Positional statistics per era versus Beta(k,6-k)                                        |')
    print(f'+-----------------------------------------------------------------------------------------+')
    print(f'|Position    |   KL   |   TV   |  X2-p  |   W   |   B   |   H   |   JS  |  KS-D  |  KS-p  |')
    num_chip_significant = 0
    ks_d_max_i = -1
    for i in range(len(lotto_config.cols)) :
        col = lotto_config.cols[i]
        if chi_p[i] < 0.05:
            num_chip_significant += 1
        if i < len(lotto_config.cols)-1:
            ks = ks_results[i]
            ks_d = ks[1]
            ks_p = ks[2]
            if ks_d > D_max:
                ks_d_max_i = i
                D_max = ks_d
            if ks_p > P_max:
                P_max = ks_p
            sig = "Yes" if ks_p < 0.05 else " No"
            print(f'| {col:>9}  | {kl_pos[i]:6.4f} | {tv_pos[i]:6.4f} | {chi_p[i]:6.4f} | {metrics[col]["W"]:5.3f} | {metrics[col]["B"]:5.3f} | {metrics[col]["H"]:5.3f} | {metrics[col]["JS"]:5.3f} | {ks_d:6.3f} | {ks_p:6.3f} | Significant? {sig}')
        else :
            print(f'| {col:>9}  | {kl_pos[i]:6.4f} | {tv_pos[i]:6.4f} | {chi_p[i]:6.4f} | {metrics[col]["W"]:5.3f} | {metrics[col]["B"]:5.3f} | {metrics[col]["H"]:5.3f} | {metrics[col]["JS"]:5.3f} | ------ | ------ |')
    print(f'+-----------------------------------------------------------------------------------------+')

    # print a summary table
    T = era.T
    alpha = 0.05

    delta_max = ls.margin_error(T=T,alpha=alpha, p=P_max)
    print(f"| Delta @ (max) p={P_max}, T={T}:", delta_max)  # ≈0.0196
    delta_p3 = ls.margin_error(T=T,alpha=alpha, p=0.03)
    print(f"| Delta @ p=0.03, T={T}:",  delta_p3)   # ≈0.0120
    KS_power_max = ls.ks_power(T=T,alpha=alpha, delta=D_max)
    print(f"| Power @ Delta={D_max:1.4f} (max), T={T}: {KS_power_max:1.4f}")  # ≈0.79
    print(f"| Power @ Delta=0.05, T={T}:",  ls.ks_power(T=T,alpha=alpha, delta=0.05))   # ≈0.02

    print(f'| (B) ... the power to detect a KS deviation of {D_max:1.4f} is {KS_power_max*100.0:2.2f}%')
    print(f'| and the estimated probabilities in this analysis will be within +/-{delta_p3*100.0:2.2f}% - +/-{delta_max*100.0:2.2f}% at 95% confidence')
    K_beta = np.sqrt(T*D_max**2)-1.36
    zbeta = np.exp(-2*K_beta**2)
    print(f"| Beta @ Delta={D_max:1.4f}, T={T}:", zbeta, f'{(1-zbeta)*100.0:2.2f}%')  # ≈0.79    #
    print(f'+=========================================================================================+')
    # Summary stastistics
    #
    max_tv_i = np.argmax(tv_pos)
    max_tv = tv_pos[max_tv_i]
    max_tv_pos = lotto_config.cols[max_tv_i]
    print(f'Positional statistics of the {era.era_string} era versus the Beta(k,6-k) theoretical. {era.T} tickets')
    chip_significant = f'{num_chip_significant} of {len(lotto_config.cols)-1} positions were significant at p<0.05' if num_chip_significant > 0 else 'No positions were significant at p<0.05'
    chip_match = 'good' if num_chip_significant < 3 else 'moderate'
    print(f'were used in this era. {chip_significant} signifying a {chip_match} match with the reference Beta function.')
    print(f'Despite having so many samples, the power to detect the deviation in {lotto_config.cols[ks_d_max_i]}')
    print(f'(KS D={D_max:1.4f}, KS-p {ks_results[ks_d_max_i][2]:1.4f} at 95%) was {KS_power_max*100.0:0.2f}%.')
    print(f'+-----------------------------------------------------------------------------------------+')
    print(f'The ball draw positions in {era.lotto} as compared to a Beta(k,6-k) function.')
    print(f'Deviations from the Beta are highlighted in filled-red areas, with the largest')
    print(f'deviation marked with a red arrow. The mean and first standard deviation are marked')
    print(f'in vertical lines. The Total Variation (TV) and KL distance (KL) are shown inset, ')
    print(f'with highest TV={max_tv:1.4f} in {max_tv_pos} with a KL-D of {kl_pos[max_tv_i]}.')

    print()
    cols_sub_p_05 = []
    for i in range(len(lotto_config.cols)-1):
        col = lotto_config.cols[pos]
        if chi_p[i] <= 0.05:
            cols_sub_p_05.append(lotto_config.cols[i])
    print('Goodness-of-fit and divergence metrics for each Powerball draw position against the ')
    print(f'theoretical Beta(k,6-k) order-statistic PMF. X2 tests all have p>0.05 except for {cols_sub_p_05}; ')
    print('Total-Variation (TV) and Kullback-Leibler (KL) quantify the magnitude of any residual differences.')
    print(f'+=========================================================================================+')

def beta_graph(era:pb.PowerballEra) -> None:
    '''
    Generates the beta and pmf graph overlays for a specific era
    :param era: The PowerballEra object containing the draw data
    '''
    lotto_config = era.config
    pxx, xxx = ls.compute_pmf(df_data=era.details,lotto_config=era.config)
    pbeta, _ = ls.compute_beta(df_data=era.details,lotto_config=era.config)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(
        f"{era.title} White-Ball Draw Positions: Empirical PMF vs. Theoretical $\\beta$(k,6-k) {era.era_string}",
        fontsize=14, y=1.02
    )
    lotto_config = era.config
    kl_pos = era.kl_by_position
    tv_pos = era.tv_by_position
    chi_p = era.chi_p_by_position
    # kl_merged_pos and chi_pos were not used in the original code, so we will not use them here.
    for pos in range(len(lotto_config.cols)-1):
        col = lotto_config.cols[pos]
        n = lotto_config.maxval[pos]
        probs = pxx[:n, pos]
        theo_beta = None
        if(pos < pbeta.shape[1]):
            theo_beta = pbeta[:n, pos]

        ax = axes[pos//2, pos%2]
        # --- truncated-normal (blue) ---
        #ax.plot(probs, label=f'Truncated {col}', color='C0')
        ax.set_xlabel("Position Value")
        ax.set_ylabel("Probability")
        ax.grid(True, which='both', axis='x', color='lightgray', linestyle='--', alpha=0.5)

        # title with max-E(p)
        amax = np.argmax(probs)
        pmax = probs[amax]
        rmax = round(pmax*100.0, 2)
        # ax.set_title(f"E(p) of {lotto} Position {col}\nMax E(p) {rmax}% at {amax+1}")
        k=pos+1
        ax.set_title(
            f"Position D{pos+1}: Empirical PMF vs. Beta({k},{6-k})\n"
            f"(Max PMF {rmax}% at {amax+1})"
        )
        # secondary axis for empirical PMF, beta & discrete
        # axx = ax.twinx()
        axx = ax
        axx.plot(probs, label=f'Empirical PMF {col}', color='gray', alpha=0.6)
        if(theo_beta is not None):
            axx.plot(theo_beta, label=f'Beta PDF {col}', color='orange', alpha=0.4)
        
        # --- exact discrete PMF of the k-th order stat ---
        if pos < 5:
            k = pos+1
            M = n
            N = 5
            x = np.arange(1, M+1)
            discrete = comb(x-1, k-1) * comb(M-x, N-k) / comb(M, N)
            axx.plot(discrete, '--', label=f'Order-stat PMF {col}', color='green', alpha=0.7)

        # shade where empirical exceeds theory
        if(theo_beta is not None):
            diff = probs - theo_beta
            x = np.arange(0, n)
            axx.fill_between(x, theo_beta, probs,
                            where=(diff>0),
                            color='red', alpha=0.2,
                            interpolate=True)

            # annotate single largest deviation
            maxi = np.argmax(diff)
            axx.annotate(
                '',
                xy=(maxi, probs[maxi]),
                xytext=(maxi, theo_beta[maxi]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
            )
        
        # mean ± std vertical lines
        mean_x, std_x = xxx[pos, 0], xxx[pos, 1]
        axx.axvline(mean_x, color='r')
        axx.axvline(mean_x+std_x, color='r', linestyle=':')
        if mean_x - std_x > 1:
            axx.axvline(mean_x-std_x, color='r', linestyle=':')

        # add light grid on y-axis
        axx.grid(True, which='both', axis='y', color='lightgray', linestyle='--', alpha=0.5)

        tv = tv_pos[pos]
        kl = kl_pos[pos]
        coords = ((.5,.9), (.6, .9), (.9, .9), (.2, .5), (.2, .9))
        axx.text(coords[pos][0], coords[pos][1],
            f"TV={tv:.3f}\nKL={kl:.3f}\n$p$={chi_p[pos]:.3f}",
            transform=axx.transAxes,
            ha='right', va='top',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6)
        )

        # inset zoom for D1 (pos=0) and D5 (pos=4)
        if pos in (0, 4):
            sloc = 'lower right' if pos == 0 else 'lower left'

            axins = inset_axes(axx, width="30%", height="30%", loc=sloc, borderpad=2)
            axins.plot(x, probs, color='gray', alpha=0.6)
            if theo_beta is not None:
                axins.plot(x, theo_beta, color='orange', alpha=0.4)
            if pos == 0:
                axins.set_xlim(1, 10)
            else:
                axins.set_xlim(n-9, n)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title("Perimeter zoom", fontsize=8)

        # legends
        if pos == 0 or pos == 1:
            axloc = 'upper right'
        elif(pos == 4):
            axloc = 'upper center'
        elif(pos == 2 or pos == 3) :
            axloc = 'upper left'
        ax.legend(loc=axloc)
        axx.legend(loc=axloc)
    fig.delaxes(axes[2,1])
    plt.tight_layout()
    out = os.path.join(era.lotto, f"{era.lotto}-beta-pmf-{era.era_string}.png")
    fig.savefig(out, facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)


def era_comparison(era1:pb.PowerballEra, era2:pb.PowerballEra) -> tuple:
    '''
    Basic era comparison looking for statistical significance between the two eras and 
    returning the comparison statistics.
    :param era1: The first PowerballEra object to compare
    :param era2: The second PowerballEra object to compare
    :return: A tuple containing the KS statistic, KS p-value, Chi-squared statistic, Chi-squared p-value, degrees of freedom,
             Bhattacharyya distance, and Wasserstein distance.
    '''
    df_era1 = era1.details
    df_era2 = era2.details
    nll_era1 = era1.NLL().to_numpy().astype(np.float64)
    nll_era2 = era2.NLL().to_numpy().astype(np.float64)
    print(f"Comparing eras: Era 1={era1.era_string} and Era 2={era2.era_string}")
    print(f"Era 1: {df_era1.shape[0]} draws, Era 2: {df_era2.shape[0]} draws")
    # KS between two eras’ NLL:
    ks_stat, ks_p = ks_2samp(nll_era1, nll_era2)
    # χ² on histogram bins of NLL:
    counts_era1, bins = np.histogram(nll_era1, bins=50)
    counts_era2, _    = np.histogram(nll_era2, bins=bins)
    B_dist = ls.bhattacharyya_distance(counts_era1, counts_era2)
    W_dist = wasserstein_distance(counts_era1, counts_era2)
    # 2) stack into a 2×50 contingency table
    table = np.vstack([counts_era1+ 1e-10, counts_era2+ 1e-10])
    # 3) run the two‐way chi‐square test
    chi2_stat, p_val, dof, expected = chi2_contingency(table)
    # chi_stat, chi_p = chisquare(f_obs=counts1, f_exp=counts2 + 1e-10)
    print(f"KS test between eras: statistic={ks_stat}, p-value={ks_p},X2 test: statistic={chi2_stat}, p-value={p_val}, dof={dof}")
    return ks_stat, ks_p, chi2_stat, p_val, dof,B_dist,W_dist

def deep_era_comparison(era1:pb.PowerballEra, era2:pb.PowerballEra,lotto:str) -> None:
    '''
    Perform the deep era comparison with graph and Bhattacharyya and Wasserstein distance calculations.
    The generated graph is the overlay histogram comparing the NLL distributions of the two eras.
    :param era1: The first PowerballEra object to compare
    :param era2: The second PowerballEra object to compare
    :param lotto: The lottery name, used for saving the graph
    '''
    nll_era1 = era1.NLL().to_numpy().astype(np.float64)
    nll_era2 = era2.NLL().to_numpy().astype(np.float64)
    era1_string = era1.era_string
    era2_string = era2.era_string
    print("===================================================================")
    print(f"Deep era comparison: Era 1={era1_string} and Era 2={era2_string}")
    # (Assume nll_pre2015, nll_post2015 are the arrays of NLLBeta for each jackpot era)
    counts_pre,  bins = np.histogram(nll_era1,  bins=50)
    counts_post, _   = np.histogram(nll_era2, bins=bins)

    # Build 2×50 table
    table = np.vstack([counts_pre+ 1e-10, counts_post+ 1e-10])
    chi2_stat, p_val, dof, expected = chi2_contingency(table)

    # Standardized residuals: (observed - expected) / sqrt(expected)
    std_resid = (table - expected) / np.sqrt(expected)

    # std_resid is a 2×50 array; 
    # row 0 = residuals for era1, row 1 = residuals for era2.

    # Find which bins have |residual| > 2 for significance
    sig_bins_era1  = np.where(np.abs(std_resid[0]) > 2)[0]
    sig_bins_era2 = np.where(np.abs(std_resid[1]) > 2)[0]
    print(f"Significant bins for Era 1 ({era1_string}): {sig_bins_era1}")
    print(f"Significant bins for Era 2 ({era2_string}): {sig_bins_era2}")

    B_dist = ls.bhattacharyya_distance(counts_pre, counts_post)
    print(f"Bhattacharyya distance between eras: {B_dist}")
    W_dist = wasserstein_distance(counts_pre, counts_post)
    print(f"Wasserstein distance between eras: {W_dist}")

    for arr, label in [(nll_era1, era1_string), (nll_era2, era2_string)]:
        print(label, 
            " mean=",  np.mean(arr),
            " var=",   np.var(arr, ddof=1),
            " skew=", pd.Series(arr).skew(),
            " kurt=", pd.Series(arr).kurtosis())
    
    fig,axes = plt.subplots(figsize=(10,5))
    axes.text(.8, .6,
        f"B={B_dist:.3f}\nW={W_dist:.3f}",
        transform=axes.transAxes,
        ha='right', va='top',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6)
    )
    _,_,patches1 = plt.hist(nll_era1,  bins=bins, alpha=0.4, label=f"{era1_string}", density=True)
    plt.xlabel("NLL")
    plt.ylabel("Density")
    _,_,patches2 = plt.hist(nll_era2, bins=bins, alpha=0.4, label=f"{era2_string}", density=True)
    highlight_patch = Patch(color='red', label='Bins with |residual| > 2$\\sigma$')
    l1=plt.axvline(x=np.mean(nll_era1), color='blue', linestyle='--', label=f"{era1_string} mean")
    l2=plt.axvline(x=np.mean(nll_era2), color='orange', linestyle='--', label=f"{era2_string} mean")
    plt.legend(handles=[patches1[0], patches2[0], highlight_patch,l1,l2])
    if len(sig_bins_era1) > 0 :
        for i in sig_bins_era1:
            # patches is the list of rectangles created by the histogram
            # Set the face color of the significant bins to red
            patches1[i].set_facecolor('red')
            patches1[i].set_alpha(0.6)
    if len(sig_bins_era2) > 0 :
        for i in sig_bins_era2:
            patches2[i].set_facecolor('red')
            patches2[i].set_alpha(0.6)
    plt.title(f"Overlay: NLL distributions {era1_string} vs. {era2_string}\n$X^2$ test: p={p_val:.3f}, $-log(p)$={-np.log(p_val):.3f}, dof={dof}")
    plt.grid(True)
    fig.tight_layout()
    fig_filename = f"nllbeta_distribution_{era1_string}_vs_{era2_string}.png"
    fig.savefig(os.path.join(lotto,fig_filename), facecolor=fig.get_facecolor())
    plt.close(fig)

def analyze_eras(eras) :
    """
    Analyze the eras and compare them.
    This function takes a list of eras and compares them using various statistical methods.
    """
    if len(eras) < 2:
        print("Not enough eras to compare.")
        return
    print("Comparing eras:")
    df_era_results = pd.DataFrame(columns=['Era1','Era2', 'KS_stat', 'KS_p', 'Chi2_stat', 'Chi2_p', 'dof', 'B','W'])
    l = []
    for i in range(len(eras)):
        # do i+1 because we only need to compare against the ones that have
        # not been compared yet.
        for j in range(i+1,len(eras)):
            if i != j: # sanity check
                print(f"Comparing era {i+1} to era {j+1}")
                ks_stat, ks_p, chi2_stat, p_val, dof,B,W = era_comparison(eras[i], eras[j])
                df_era_results.loc[len(df_era_results)] = {
                    'Era1': f'{eras[i].era_string}',
                    'Era2': f'{eras[j].era_string}',
                    'KS_stat': ks_stat,
                    'KS_p': ks_p,
                    'Chi2_stat': chi2_stat,
                    'Chi2_p': p_val,
                    'dof': dof,
                    'B': B,
                    'W': W
                }
                l.append(f"| {eras[i].era_string:<25} | {eras[j].era_string:<25} | {ks_stat:<7.3f} | {ks_p:<6.3f} | {chi2_stat:<9.3f} | {p_val:<6.3f} | {dof:<4d} | {B:<4.3f} | {W:<4.3f} +")
                if p_val < 0.05:
                    print(f"Significant difference between Era {i+1} {eras[i].era_string} and Era {j+1} {eras[j].era_string} (p-value={p_val})")
                    deep_era_comparison(eras[i], eras[j], eras[i].lotto)
    print("+---------------------------------------------------------------------------------------------------------------------+")
    print("+ Era1                      | Era2                      | KS_stat | KS_p   | Chi2_stat | Chi2_p | dof  |  B    |   W  +")
    for s in l :
        print(s)
    print("+---------------------------------------------------------------------------------------------------------------------+")
    return df_era_results

def plot_nll_with_burstiness(era:pb.PowerballEra, ax:plt.Axes, window_days:int=90,cmap=None, norm=None)-> PathCollection:
    '''
    Plot the NLL graph with burstiness coloring.
    :param era: The PowerballEra object containing the draw data
    :param ax: The matplotlib axes to plot on
    :param window_days: The number of days to use for the burstiness calculation
    :param cmap: The colormap to use for coloring the lines
    :param norm: The normalization to use for the colormap
    returns the path collection of the scatter plot for the jackpot wins.
    '''
    x = era.details['Offset']
    y = era.details['NLLbeta']
    c = era.details['running_burstiness']
    
    # Build line segments: shape (n−1, 2, 2)
    points   = np.vstack([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize colors to the 5–95 percentile band
    if cmap is None or norm is None:
        vmin, vmax = np.nanpercentile(c, [5,95])
        cmap, norm = mpl.cm.berlin, mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create the collection and add to axes
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(c)                    # color each segment by the burstiness at its start
    ax.add_collection(lc)

    # Overlay jackpot‐win arrows or markers
    mask = era.details['is_jackpot'] == 1
    x_jack = era.details.loc[mask, 'Offset']
    y_jack = era.details.loc[mask, 'NLLbeta'].values

    # Option A: simple triangle markers
    im = ax.scatter(x_jack, y_jack,
               marker='v',    # downward triangle
               color='black',
               s=30,
               zorder=5,
               label='Jackpot win')
    
    # Fix the view limits
    ax.set_xlim(x.min()-3, x.max()+3)
    ax.set_ylim(y.min()-.1, y.max()+.1)

    ax.set_xlabel("Days since era start")
    ax.set_ylabel("NLL")
    ax.set_title(f"NLL colored by {window_days}d burstiness")
    # Add a colorbar
    ax.figure.colorbar(lc, ax=ax, pad=0.01).set_label("Burstiness")
    return im

def plot_logw_with_burstiness(era:pb.PowerballEra, ax:plt.Axes, window_days:int=90,cmap=None, norm=None)->PathCollection:
    '''
    Plots the log10(Winners) graph with burstiness coloring.
    :param era: The PowerballEra object containing the draw data
    :param ax: The matplotlib axes to plot on
    :param window_days: The number of days to use for the burstiness calculation
    :param cmap: The colormap to use for coloring the lines
    :param norm: The normalization to use for the colormap
    returns the path collection of the scatter plot for the jackpot wins.
    '''
    x = era.details['Offset']
    y = era.winners_log10
    c = era.details['running_burstiness']
    
    points   = np.vstack([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if cmap is None or norm is None:
        vmin, vmax = np.nanpercentile(c, [5,95])
        cmap, norm = mpl.cm.berlin, mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(c)
    ax.add_collection(lc)

    # then overlay the wins
    mask = era.details['is_jackpot'] == 1
    x_jack = era.details.loc[mask, 'Offset']
    y_jack = np.log10(era.details.loc[mask, 'Winners'].to_numpy().astype(np.float64))

    im = ax.scatter(x_jack, y_jack,
               marker='v',
               color='black',
               s=30,
               zorder=5)

    ax.set_xlim(x.min()-3, x.max()+3)
    ax.set_ylim(y.min()-.1, y.max()+.1)

    ax.set_xlabel("Days since era start")
    ax.set_ylabel("log10(Winners)")
    ax.set_title(f"log10(Winners) colored by {window_days}d burstiness")
    ax.figure.colorbar(lc, ax=ax, pad=0.01).set_label("Burstiness")
    return im

def plot_nll_and_logw_with_burstiness(era:pb.PowerballEra, window_days:int=90)->None:
    '''
    Plot the NLL and log10(Winners) graphs with burstiness coloring.
    :param era: The PowerballEra object containing the draw data
    :param window_days: The number of days to use for the burstiness calculation
    This function creates two subplots: one for NLL and one for log10(Winners), both colored by burstiness.
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # Normalize colors to the 5–95 percentile band
    dates = era.details['Date']  # datetime64[ns] Series
    c = era.details['running_burstiness'].to_numpy().astype(np.float64)
    vmin, vmax = np.nanpercentile(c, [5,95])
    cmap, norm = mpl.cm.berlin, mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    _ = plot_nll_with_burstiness(era, ax1, window_days=window_days,cmap=cmap, norm=norm)
    _ = plot_logw_with_burstiness(era, ax2, window_days=window_days,cmap=cmap, norm=norm)
    ax1.set_xlabel('')
    ax1.xaxis.set_ticklabels([])
    ax2.set_xlabel(f"Historical Ticket in Draws Before End of Era, B={era.jackpot_burstiness:.3f}, FI={era.jackpot_fano:.3f}")
    tick_pos = np.linspace(0, dates.shape[0]-1, 8, dtype=int)
    tick_labels = [dates.iloc[i].strftime("%Y\n%m-%d") for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels) #, rotation=45, ha='right')
    plt.suptitle(f"{era.title} NLL and log10(Winners) with {window_days}d Burstiness in {era.era_string}", fontsize=16)
    plt.tight_layout()
    fig_filename = f"{era.lotto}-NLL-bursti-{era.era_string}-winning-tickets.png"
    fig.savefig(os.path.join(era.lotto,fig_filename), facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)

def plot_nll(era:pb.PowerballEra, ax:plt.Axes) -> None :
    '''
    Plot the NLL graph for the winning tickets in the era.
    :param era: The PowerballEra object containing the draw data
    :param ax: The matplotlib axes to plot on
    This function plots the NLL of the winning tickets in the era and annotates the jackpot wins.
    '''
    ymax1 = era.details['NLLbeta'].max()
    seq = np.arange(1,era.T,3)
    ax.plot(era.details['NLLbeta'],label='NLL of Ticket')
    ax.set_title(f'{era.title} Negative Log Likelihood for Winning Tickets in {era.era_string}')
    ax.set_ylabel("NLL")
    for xx,yy1,yy2,_,_,yy3 in era.annotations :
        ymax2_ = ymax1
        if ymax1-yy3 < 0.01:
            ymax2_ += 0.25
        ax.annotate('', xy=(xx,yy3), xycoords='data',xytext=(xx,ymax2_), arrowprops={"arrowstyle":"->","connectionstyle":"arc3","facecolor":"black"})

def plot_logw(era:pb.PowerballEra, ax:plt.Axes) -> None :
    '''
    Plot the log10(Winners) graph for the winning tickets in the era.
    :param era: The PowerballEra object containing the draw data
    :param ax: The matplotlib axes to plot on
    This function plots the log10(Winners) of the winning tickets in the era and annotates the jackpot wins.
    '''
    ymax2 = era.winners_log10.max()
    ax.set_title('Log Count of Winners Per Ticket')
    ax.plot(era.winners_log10,color='r')
    ax.set_ylabel(f'$log_10$(Winners)')
    for xx,yy1,yy2,_,_,yy3 in era.annotations :
        ymax2_ = ymax2
        if ymax2-yy2 < 0.01:
            ymax2_ += 0.25
        ax.annotate('', xy=(xx,yy2), xycoords='data',xytext=(xx,ymax2_), arrowprops={"arrowstyle":"->","connectionstyle":"arc3","facecolor":"black"})

def plot_jackpot_to_jackpot_all_eras(eras:list[pb.PowerballEra]) -> None:
    '''
    Plot a single jackpot-to-jackpot interval from each era in it sown
    plot panel using log10(winners) as the y-axis.
    :param eras: A list of PowerballEra objects to plot
    This function creates a 2x2 grid of subplots, each showing the jackpot-to-jackpot interval for a different era.
    Each subplot shows the log10(Winners) for the jackpot-to-jackpot intervals, with the jackpot wins marked in red.
    The x-axis is the date of the draw, and the y-axis is log10(Winners).
    The x-axis is inverted to show the most recent draws on the left.
    '''
    fig, axes = plt.subplots(2,2,figsize=(14, 14))
    fig.suptitle("Jackpot-to-Jackpot Focus interval For Each Era. Red Dot Denotes Jackpot Win", fontsize=16)
    panel = 0
    for era in eras :
        details = era.details.sort_values(by='Offset',ascending=True)
        mask = details['is_jackpot'] == 1
        jp_winners = details.loc[mask, 'Winners']
        jp_dates = details.loc[mask, 'Date']
        offsets = details.loc[mask, 'Offset']
        for i in range(1,len(offsets)):
            mask2 = details['Offset'].between(offsets.iloc[i-1], offsets.iloc[i])
            winners_sum = details.loc[mask2,'Winners'].sum()
            if winners_sum > 8000000:  # only plot if winners are significant
                print(f"Plotting jackpot-to-jackpot interval {i} in {era.era_string} with {winners_sum} winners")
                mask2 = details['Offset'].between(offsets.iloc[i-1], offsets.iloc[i],inclusive='both')
                dates = details.loc[mask2, 'Date']
                ax = axes[panel//2, panel%2]
                ax.set_title(f"{era.era_string} - {winners_sum:,} Winners")
                ax.set_xlabel("Days")
                ax.set_ylabel("log10(Winners)")
                winners = details.loc[mask2, 'Winners'].to_numpy().astype(np.float64)
                log_winners = np.log10(winners)
                ax.plot(dates, log_winners, marker='o', label=f"Jackpot {i}")
                im = ax.scatter(jp_dates.loc[mask2], np.log10(jp_winners.loc[mask2].to_numpy().astype(np.float64)),
                        marker='o',
                        color='red',
                        s=30,
                        zorder=5)
                ax.invert_xaxis()  # invert x-axis to show most recent on the left
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
                panel += 1
                break
    plt.tight_layout()
    fig_filename = f"{era.lotto}-jackpot-to-jackpot-all-eras.png"
    fig.savefig(os.path.join(era.lotto,fig_filename), facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)

def plot_pretty_winners_between_jackpots_with_nll(era:pb.PowerballEra) -> None:
    '''
    Plot the winners between jackpots with NLL coloring.
    :param era: The PowerballEra object containing the draw data
    This function creates a plot showing the log10(Winners) between jackpots, colored by NLL.
    It includes a top panel with a line plot of log10(Winners) at each jackpot, and a bottom panel with a histogram of those values.
    The x-axis is the date of the jackpot win, and the y-axis is log10(Winners).
    The x-axis is inverted to show the most recent draws on the left. 
    It also includes a colorbar for the NLL values.
    '''
    # Prep the DataFrame
    df = era.details.copy()
    jack_df = df[df['is_jackpot'] == 1]

    # Compute summary stats on the jackpot‐to‐jackpot log10 winners
    vals = jack_df['Winners_between_jackpots_log10']
    mean_val   = vals.mean()
    median_val = vals.median()
    std_val    = vals.std(ddof=1)

    # now color each dot by NLLbeta
    cvals = jack_df['NLLbeta'].values
    vmin, vmax = np.nanpercentile(cvals, [5,95])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps['plasma']

    # Styling

    # Make the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=False)
    fig.suptitle(
        f"Winners Between Jackpots in {era.era_string}\n"
        f"Burstiness $B$={era.jackpot_burstiness:.3f}, "
        f"Fano $FI$={era.jackpot_fano:.3f}",
        fontsize=18,
        y=0.97
    )

    # ─── Top panel: line of log10 winners at each jackpot ────────────────────────
    #    marker='o',
    #    markersize=6,
    ax1.plot(
        jack_df['Date'],
        vals,
        linestyle='-',
        linewidth=2,
        label='Log$_{10}$(Winners Between Jackpots)'
    )
    sc = ax1.scatter(
        jack_df['Date'],
        vals,
        c=cvals,
        cmap=cmap,
        norm=norm,
        s=80,
        edgecolor='k',
        linewidth=0.5,
        label='NLL at Jackpot'
    )
    ax1.set_ylabel('Log$_{10}$(Winners Between Jackpots)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_title("Each Jackpot's Total Winners Since Previous Win", fontsize=16)
    ax1.axhline(mean_val,   color='C1', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax1.axhline(median_val, color='C2', linestyle=':',  linewidth=2, label=f"Median = {median_val:.2f}")

    cbar = fig.colorbar(sc, ax=ax1, pad=0.01,fraction=0.03)
    cbar.set_label('NLL', fontsize=12)

    # format dates on the top panel but hide tick labels
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    ax1.invert_xaxis()  # invert x-axis to show most recent on the left
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax1.set_xlabel(f"Jackpot Win Date")

    # ─── Bottom panel: histogram of those values ────────────────────────────────
    n_bins = 30
    ax2.hist(
        vals, 
        bins=n_bins,
        alpha=0.6,
        edgecolor='black',
        label='Winners Between Jackpots'
    )
    # vertical lines for mean & median
    ax2.axvline(mean_val,   color='C1', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax2.axvline(median_val, color='C2', linestyle=':',  linewidth=2, label=f"Median = {median_val:.2f}")

    ax2.set_xlabel('Log$_{10}$(Winners Between Jackpots)', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title("Distribution of Winners Between Jackpots", fontsize=16)
    ax2.legend(loc='upper left', fontsize=12)

    # share the same date axis only for the bottom (if you prefer calendar ticks)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Tight layout & save
    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(era.lotto, f"{era.lotto}-Winners-Between-Jackpots-nll-{era.era_string}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def summarize_era(era:pb.PowerballEra) -> None:
    '''
    Summarize the jackpot analysis for a specific era.
    :param era: The PowerballEra object containing the draw data
    This function prints the jackpot analysis for the era, including the NLL beta statistics,
    Bhattacharyya distance, Wasserstein distance, jackpot records, and burstiness.
    It also plots the NLL and log10(Winners) graphs with burstiness coloring.
    '''
    print("====================================================================================")
    print(f"Era: {era.era_string}, jackpot analysis for {era.jackpots.shape[0]} jackpot win draws")
    nll = era.details['NLLbeta']
    print(f'Min/Max/Mean of NLL beta: {nll.min()}, {nll.max()}, {nll.mean()}')
    jnll = era.details['jNLL']
    B = ls.bhattacharyya_distance(nll, jnll)
    print(f'Bhattacharyya distance between the empirical NLL and the joint NLL = {B}')
    W = wasserstein_distance(nll, jnll)
    print(f'Wasserstein distance between the empirical NLL and the joint NLL = {W}')
    print("Jackpot records in this split:")
    print(era.jackpots.describe())
    nll = era.jackpots['NLLbeta']
    print(f'Min/Max/Mean of NLL beta: {nll.min()}, {nll.max()}, {nll.mean()}')
    rec = era.max_jackpot
    print("Max NLL_beta record")
    print(rec)
    #
    # spearman correlation of the NLL to the winners
    a = era.details['NLLbeta']
    b = era.winners_log10
    rho,p = stats.spearmanr(a,b)
    print(f'{era.title} {era.era_string} : NLL to Winners : Spearman rho={rho}, p={p}')
    rho,p = stats.pearsonr(a.to_numpy().astype(np.float64),b.astype(np.float64))
    print(f'{era.title} {era.era_string} : NLL to Winners : Pearson rho={rho},p={p}')

    a = era.details['NLLbeta']/era.details['jNLL']
    rho,p = stats.spearmanr(a,b)
    print(f'{era.title} {era.era_string} : NLL/jNLL to Winners : Spearman rho={rho}, p={p}')
    rho,p = stats.pearsonr(a.to_numpy().astype(np.float64),b.astype(np.float64))
    print(f'{era.title} {era.era_string} : NLL/jNLL to Winners : Pearson rho={rho},p={p}')
    E_a = a.mean()
    E_b = b.mean()
    ab = a * b
    E_ab = ab.mean()
    cov = E_ab - E_a*E_b
    std_a = a.std()
    std_b = b.std()
    r_s = cov / (std_a * std_b)
    print(f'{era.title} {era.era_string} : NLL/jNLL to Winners : rs={r_s} (should match pearson)')
    #
    # divergence
    a_h = a / a.sum()
    b_h = b / b.sum()
    ab_h = a_h - b_h
    ab_h2 = ab_h*ab_h
    div = np.sqrt(ab_h2.max())
    print(f'{era.title} {era.era_string} : NLL/jNLL to Winners : Divergence={div}')
    print(f'{era.title} {era.era_string} : Burstiness of jackpot wins: {era.jackpot_burstiness:.3f}')
    fano = era.jackpot_fano
    if era.jackpot_burstiness < -0.5 :
        print("This era has low burstiness, indicating anti-clustering and a more regular cadence of jackpot wins.")
    elif era.jackpot_burstiness > 0.5 :
        print("This era has high burstiness, indicating clustering and a more irregular cadence of jackpot wins.")
    else :
        print("This era has moderate burstiness, indicating an expected Poisson-like distribution of jackpot wins.")
    if fano < 0.5 :
        print("This era has low Fano factor, indicating a more regular cadence of jackpot wins.")
    elif fano >= 1.5 :
        print("This era has high Fano factor, the jackpot cadence is over-dispersed (clustered).")
    else :
        print("This era has moderate Fano factor, indicating an expected Poisson-like distribution of jackpot wins.")
    fig,axes = plt.subplots(2,1,figsize=(16,12))
    plot_nll(era, axes[0])
    plot_logw(era, axes[1])
    axes[0].legend()
    seq = np.arange(1,era.winners_log10.shape[0],3)
    axes[0].set_xticks(seq,minor=True)
    axes[1].set_xticks(seq,minor=True)
    axes[1].set_xlabel(f"Historical Ticket in Days Before End of Era, B={era.jackpot_burstiness:.3f}, FI={fano:.3f}")
    fig.tight_layout()
    fig_filename = f"{era.lotto}-NLL-{era.era_string}-winning-tickets.png"
    fig.savefig(os.path.join(era.lotto,fig_filename), facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)
    #
    # Setup the winners between jackpots for cadence analysis and plotting
    #
    era.details['grp'] = era.details['is_jackpot'].astype(int).cumsum()
    era.details['grp_adj'] = era.details['grp'] - era.details['is_jackpot']
    interval_sums = era.details.groupby('grp_adj')['Winners'].sum().rename('Winners_between_jackpots')
    era.details = era.details.join(interval_sums, on='grp_adj')
    era.details['Winners_between_jackpots'] = era.details['Winners_between_jackpots'].fillna(0)
    era.details['Winners_between_jackpots_log10'] = np.log10(era.details['Winners_between_jackpots'].replace(0, np.nan))
    era.details['Winners_between_jackpots_log10'] = era.details['Winners_between_jackpots_log10'].fillna(0)

    plot_nll_and_logw_with_burstiness(era, window_days=90)
    plot_pretty_winners_between_jackpots_with_nll(era)

def summarize(input_file_name: str, lotto: str) -> None:
    """
    Summarize the jackpot history data from a CSV file.
    This function reads the jackpot data and prints basic statistics.
    :param input_file_name: The name of the CSV file containing the jackpot history
    :param lotto: The name of the lottery (e.g., 'PowerBall')
    :return: None
    1. It checks if the file exists.
    2. It reads the CSV file into a DataFrame.
    3. It prints the jackpot history statistics, including the number of draws, columns,
    and the maximum NLL_beta record.
    4. It calculates and prints the Bhattacharyya and Wasserstein distances between the
    empirical NLL and the joint NLL.
    5. It summarizes the jackpot records and prints the top and bottom 5 records.
    6. It checks for time splits in the lottery configuration and summarizes each era.
    7. It compares eras if there are multiple time splits.
    :raises FileNotFoundError: If the specified jackpot file does not exist.
    :raises ValueError: If the lottery name is not recognized.
    """
    if not os.path.exists(input_file_name):
        print(f"Error: The specified jackpot file '{input_file_name}' does not exist.")
        return
    lotto_config = lc.LottoConfig()[lotto]
    df_jackpot = pd.read_csv(input_file_name)
    print(f"Jackpot history for {lotto}:")
    print(df_jackpot.describe())
    print(f"Total draws: {df_jackpot.shape[0]}")
    print(f"Columns: {df_jackpot.columns.tolist()}")
    rec = df_jackpot[df_jackpot.NLLbeta==df_jackpot.NLLbeta.max()][lotto_config.cols+['NLLbeta','Winners','Date']]
    print("Max NLL_beta record")
    print(rec)
    sorted = df_jackpot.sort_values(by='NLLbeta', ascending=True)
    print(sorted[lotto_config.cols+['NLLbeta','Winners','Date']].head(5))
    print(sorted[lotto_config.cols+['NLLbeta','Winners','Date']].tail(5))
    #
    # jackpot only
    sorted = sorted[sorted['Jackpot'] == 'Yes']
    print("Jackpot NLL_beta records")
    print(sorted.describe())
    print(sorted[lotto_config.cols+['NLLbeta','Winners','Date']].head(5))
    print(sorted[lotto_config.cols+['NLLbeta','Winners','Date']].tail(5))
    nll = df_jackpot['NLLbeta']
    print(f'Min/Max/Mean of NLL beta: {nll.min()}, {nll.max()}, {nll.mean()}')
    jnll = df_jackpot['jNLL']
    B = ls.bhattacharyya_distance(nll, jnll)
    print(f'Bhattacharyya distance between the empirical NLL and the joint NLL = {B}')
    W = wasserstein_distance(nll, jnll)
    print(f'Wasserstein distance between the empirical NLL and the joint NLL = {W}')
    time_splits = lc.LottoConfig().lotto_splits(lotto)
    if time_splits is not None :
        eras = []
        for start_dt,end_dt in time_splits:
            print("===================================================================")
            if end_dt is None:
                print(f"Time split for {lotto} from {start_dt} to present")
                df_split = df_jackpot[df_jackpot['Date'] >= start_dt.strftime('%Y-%m-%d')]
                summarize_era(df_split, lotto_config)
            else:
                print(f"Time split for {lotto} from {start_dt} to {end_dt}")
                df_split = df_jackpot[(df_jackpot['Date'] >= start_dt.strftime('%Y-%m-%d')) & (df_jackpot['Date'] <= end_dt.strftime('%Y-%m-%d'))]
                summarize_era(df_split, lotto_config)
            era = pb.PowerballEra(df_split, 'PowerBall', lotto_config)
            eras.append(era)
        # compare eras
        if len(eras) > 1:
            print("Comparing eras:")
            l = []
            df_era_results = pd.DataFrame(columns=['Era1','Era2', 'KS_stat', 'KS_p', 'Chi2_stat', 'Chi2_p', 'dof', 'B','W'])
            for i in range(len(eras)):
                # do i+1 because we only need to compare against the ones that have
                # not been compared yet.
                for j in range(i+1,len(eras)):
                    if i != j: # sanity check
                        print(f"Comparing era {i+1} to era {j+1}")
                        ks_stat, ks_p, chi2_stat, p_val, dof,B,W = era_comparison(eras[i], eras[j])
                        df_era_results.loc[len(df_era_results)] = {
                            'Era1': f'{eras[i].era_string}',
                            'Era2': f'{eras[j].era_string}',
                            'KS_stat': ks_stat,
                            'KS_p': ks_p,
                            'Chi2_stat': chi2_stat,
                            'Chi2_p': p_val,
                            'dof': dof,
                            'B': B,
                            'W': W
                        }
                        l.append(f"| {eras[i].era_string:<25} | {eras[j].era_string:<25} |  {ks_stat:.3f}  | {ks_p:.4f} | {chi2_stat:4.4f} | {p_val:.4f} | {dof:>4} | {B:.4f} | {W:.4f} |")
                        if p_val < 0.05:
                            print(f"Significant difference between Era {i+1} {eras[i][0]}-{eras[i][1]} and Era {j+1} {eras[j][0]}-{eras[j][1]} (p-value={p_val})")
                            deep_era_comparison(eras[i], eras[j],lotto)
            print("+----------------------------------------------------------------------------------------------------------------------+")
            print("+ Era1                      | Era2                      | KS_stat | KS_p   | Chi2_stat | Chi2_p | dof  |  B  |  W      +")
            for s in l :
                print(s)
            print("+----------------------------------------------------------------------------------------------------------------------+")
            df_era_results.to_csv(os.path.join(lotto, f'{lotto}-era-comparison.csv'), index=False)
            print(df_era_results)

def plot_montecarlo_ci(era:pb.PowerballEra) -> None :
    '''
    Creates the figure that shows the Monte Carlo PMF and compares it to the empirical PMF.
    :param era: The PowerballEra object containing the draw data
    This function plots the empirical PMF against the Monte Carlo PMF for each position in the lottery ticket.
    It also shows the 95% confidence interval for the Monte Carlo PMF and highlights the outliers.
    The figure is saved as a PNG file in the era's lottery directory.
    '''
    lotto_config = era.config
    epfig, epaxes = plt.subplots(6, 2, figsize=(15, 10),gridspec_kw={'height_ratios': [3,1,3,1,3,1]})
    epfig.suptitle(
        f"Monte Carlo Uniform Random PMF {era.mc_all_pmf.shape[0]}x{era.T} Samples\nFor {era.title} Era {era.era_string}",
        fontsize=14, y=1.02
    )

    u_probs = era.mc_all_pmf[0,:,:].reshape(era.mc_all_pmf.shape[1], len(lotto_config.cols))
    emp_probs = era.pmf[0]
    u_mustd = []
    cpos = 0
    for r in range(0,6,2) :
        for xp,axes in enumerate(epaxes[r]) :
            pos = cpos + xp
            col = lotto_config.cols[pos]
            n = lotto_config.maxval[pos]
            x_vals = np.arange(1, lotto_config.maxval[pos]+1)
            # axes = epaxes[pos//2,pos%2]
            axes.plot(x_vals,u_probs[:n,pos], color='g', linestyle='-', label=f'uRandom {lotto_config.cols[pos]}')
            axes.plot(x_vals,emp_probs[:n,pos], color='r', linestyle='--',label=f'Empirical {lotto_config.cols[pos]}')
            # From chat, the simulation envelope
            axes.fill_between(x_vals, era.mc_lower[pos][:n], era.mc_upper[pos][:n], color='C3', alpha=0.2, label='95% MC envelope')
            axes.set_ylabel("Probability")
            axes.scatter(x_vals[era.mc_outliers[pos]], emp_probs[:n,pos][era.mc_outliers[pos]], color='red', zorder=3, label='Outside 95% band')
            axes.set_title(f"PMF For Position {col}")
            mu = u_probs[:,pos].mean()
            std = u_probs[:,pos].std()
            mean_x = era.pmf[1][pos,0]
            std_x = era.pmf[1][pos,1]
            u_mustd.append((mean_x,std_x))
            print(f'X: {col} mean={mean_x}, std={std_x}')
            axes.axvline(x=mean_x,color='r')
            axes.axvline(x=mean_x+std_x,color='r',linestyle=':')
            if mean_x - std_x > 1 :
                axes.axvline(x=mean_x-std_x,color='r',linestyle=':')

            wasser = era.mc_histo_metrics[col]['W']
            bhatta = era.mc_histo_metrics[col]['B']
            hellinger = era.mc_histo_metrics[col]['H']
            jensen = era.mc_histo_metrics[col]['JS']
            axes.set_xlabel(f"W={wasser:.3f} B={bhatta:.3f} H={hellinger:.3f} J={jensen:.3f}")
            seq = np.arange(1,lotto_config.maxval[5]+1,3)
            axes.set_xticks(seq,minor=True)
            axes.legend()

        for xp,axes in enumerate(epaxes[r+1]) :
            pos = cpos + xp
            col = lotto_config.cols[pos]
            n = lotto_config.maxval[pos]
            x_vals = np.arange(1, lotto_config.maxval[pos]+1)
            dd = emp_probs[:,pos] - u_probs[:,pos]
            axes.plot(x_vals,dd[:n], color='g', linestyle='-', label=f'Empirical - uRandom')
            axes.axhline(y=0, color='r', linestyle='--')
            axes.set_title("Empirical - uRandom")
            seq = np.arange(1,lotto_config.maxval[5]+1,3)
            axes.set_xticks(seq,minor=True)
        cpos += 2

    epfig.tight_layout()
    fig_filename = f"Uniform-Random-vs-Empirical-PMF-{era.era_string}.png"
    epfig.savefig(os.path.join(era.lotto,fig_filename), facecolor=epfig.get_facecolor(),bbox_inches='tight')
    plt.close(epfig)
    montecarlo_nll_ci_graph(era)

def montecarlo_nll_ci_graph(era:pb.PowerballEra) -> None :
    '''
    Creates the figure that shows the NLL of the monte carlo tickets with the 95% envelope
    and compares it to the empirical NLL.
    :param era: The PowerballEra object containing the draw data
    This function plots the Monte Carlo NLL mean and the empirical NLL for each ticket.
    It also shows the 95% confidence interval for the Monte Carlo NLL and highlights the outliers.
    The figure is saved as a PNG file in the era's lottery directory.
    '''
    print("=========================================================================")
    print(f"Monte Carlo CI for NLL vs Empirical NLL: {era.era_string}")
    print(f"Monte Carlo runs: {era.mc_all_nlls.shape[0]}, Tickets: {era.mc_all_nlls.shape[1]} (should match {era.montecarlo_history[0].shape[0]} and {era.T})")
    print("=========================================================================")
    lotto_config = era.config
    mc_nll = np.mean(era.mc_all_nlls, axis=0) # (Tickets,)
    print(f"MC NLL min/max/mean: {mc_nll.min()}/{mc_nll.max()}/{mc_nll.mean()} for {mc_nll.shape[0]} tickets")
    empirical_nll = era.details['NLLbeta']
    dates = era.details['Date']
    if mc_nll.shape[0] != empirical_nll.shape[0] :
        print(f"Error: Monte Carlo NLL shape {mc_nll.shape} does not match empirical NLL shape {empirical_nll.shape}")
        raise ValueError("Monte Carlo NLL and Empirical NLL shapes do not match")
    # should be the same size
    fig, ax = plt.subplots(figsize=(10,6))
    x_vals = np.arange(0, empirical_nll.shape[0])
    outliers = (empirical_nll < era.mc_nll_lower) | (empirical_nll > era.mc_nll_upper)

    # axes = epaxes[pos//2,pos%2]
    ax.plot(x_vals,mc_nll, color='g', linestyle='-', label=f'MC $\\mu$NLL')
    ax.plot(x_vals,empirical_nll, color='r', linestyle='--',label=f'Empirical NLL')
    # From chat, the simulation envelope
    ax.fill_between(x_vals, era.mc_nll_lower, era.mc_nll_upper, color='C3', alpha=0.2, label='95% MC envelope')
    ax.set_ylabel("Probability")
    ax.scatter(x_vals[outliers], empirical_nll[outliers], color='red', zorder=3, label='Outside 95% band')
    ax.set_title(f"Monte Carlo NLL 95% CI vs Empirical NLL For {era.title} Era {era.era_string}")
    tick_pos = np.linspace(0, dates.shape[0]-1, 8, dtype=int)
    tick_labels = [dates.iloc[i].strftime("%Y\n%m-%d") for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels) #, rotation=45, ha='right')
    ax.set_xlabel("Draw Index")
    ax.set_ylabel("NLL")
    ax.legend(loc="best")
    fig.tight_layout()
    # ax.fill_between(np.arange(len(nll)), mc_nll['lower'], mc_nll['upper'], color='orange', alpha=0.3, label='95% MC CI')
    fig_filename = f"MonteCarlo-NLL-CI-{era.era_string}.png"
    fig.savefig(os.path.join(era.lotto,fig_filename), facecolor=fig.get_facecolor(),bbox_inches='tight')
    plt.close(fig)

def montecarlo_summary_stats(eras:list[pb.PowerballEra]) -> pd.DataFrame  :
    '''
    Prints summary stats for the monte carlo run
    :param eras: A list of PowerballEra objects to analyze
    This function iterates over each era, retrieves the Monte Carlo histogram metrics,
    and prints a summary table of the Bhattacharyya, Wasserstein, Hellinger, and Jensen-Shannon distances
    for each position in the lottery ticket.
    '''
    print("=============================================================================")
    print("Monte Carlo summary statistics for eras:")
    print("+----------------------------+-----------+--------+--------+--------+--------+")
    print("| Era                        | Position  | B      | W      | H      | JS     |")
    print("+----------------------------+-----------+--------+--------+--------+--------+")
    lotto_config = eras[0].config

    df = pd.DataFrame(columns=['Era', 'Pos', 'B', 'W', 'H', 'JS'])
    i_era = 0
    for era in eras:
        if not hasattr(era, 'mc_histo_metrics'):
            print(f"Era {era.era_string} does not have Monte Carlo histogram metrics.")
            continue
        metrics = era.mc_histo_metrics
        for pos in range(len(lotto_config.cols)):
            col = lotto_config.cols[pos]
            print(f"| {era.era_string:<24}   | {col:<9} | {metrics[col]['B']:0.4f} | {metrics[col]['W']:0.4f} | {metrics[col]['H']:0.4f} | {metrics[col]['JS']:0.4f} |")
            row = {
                'Era': era.era_string,
                'Pos': col,
                'B': metrics[col]['B'],
                'W': metrics[col]['W'],
                'H': metrics[col]['H'],
                'JS': metrics[col]['JS']
            }
            df.loc[i_era] = pd.Series(row)
            i_era += 1
    print("+----------------------------+-----------+--------+--------+--------+--------+")
    return df

def powerball(args) -> None:
    # Need the dates when the game changed to get the correct lotto config
    input_file_name = os.path.join(os.path.join("..","data"),"powerball-results-1992-04-22-2025-05-30.csv")
    run_analysis(input_file_name, "powerball", mc_runs=args.mc)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## MAIN
## 
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the jackpot history analysis.')
    parser.add_argument('--csv', type=str, default=None, help='csv file to load and analyze')
    parser.add_argument('--manifest', type=str, default=None, help='manifest file of pickel dumps to load and analyze')
    parser.add_argument('--mc', type=int, default=0, help='Number of Monte Carlo simulations to run, per era, for comparative analysis')
    args = parser.parse_args()

    # Load the lottery configuration
    lotto_config = lc.LottoConfig()['powerball']
    if lotto_config is None:
        print(f"Error: No configuration found for lottery type 'powerball'")
        exit(1)
    print(f"Running jackpot history analysis for Powerball")
    if not args.csv is None:
        # If a specific CSV file is provided, load it
        if not os.path.exists(args.csv):
            print(f"Error: The specified CSV file '{args.csv}' does not exist.")
            exit(1)
        input_file_name = args.csv
        summarize(input_file_name, 'powerball')
    if not args.manifest is None:
        # If a manifest file is provided, load the pickled era data
        if not os.path.exists(args.manifest):
            print(f"Error: The specified manifest file '{args.manifest}' does not exist.")
            exit(1)
        eras = []
        with open(args.manifest, 'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                with open(line, 'rb') as era_fp:
                    era = pkl.load(era_fp)
                    eras.append(era)
                    era.report()
                    beta_graph(era)
                    pmf_graph(era)
                    qq_era_graph(era)
                    beta_graph_with_ci(era)
                    summarize_era(era)
                    plot_montecarlo_ci(era)
        df_eras = analyze_eras(eras)
        df_eras.to_csv(os.path.join('powerball','era_comparison.csv'), index=False)
        df_mc = montecarlo_summary_stats(eras)
        df_mc.to_csv(os.path.join('powerball','montecarlo_summary_stats.csv'), index=False)
        plot_jackpot_to_jackpot_all_eras(eras)
        print("Powerball history analysis completed.")
    else :
        # Load the jackpot history data and run the powerball analysis.
        powerball(args)

