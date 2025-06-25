import numpy as np
import pandas as pd
import math
from LottoConfig import LottoConfigEntry
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import beta as Beta
from scipy.special import comb
from scipy.stats import chi2
from scipy.stats import chisquare
from scipy.stats import kstest
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def merge_tail_bins(counts:list, expected:list, min_exp=5) -> tuple:
    """
    Merge low-frequency bins at both ends so that every expected[i] >= min_exp,
    but skip merging an end if the first/last 'good' bin is already at the edge.
    Returns (counts_merged, expected_merged).
    """
    M = len(expected)
    good = np.where(expected >= min_exp)[0]
    if len(good) == 0:
        raise ValueError("No bins have expected >= min_exp")

    i_low, i_high = good[0], good[-1]
    new_counts = []
    new_expected = []

    # only merge lower tail if there's actually something below the first "good" bin
    if i_low > 0:
        lower_count = counts[:i_low].sum()
        lower_exp   = expected[:i_low].sum()
        new_counts.append(lower_count)
        new_expected.append(lower_exp)

    # middle bins (all already >= min_exp)
    new_counts.extend(counts[i_low : i_high+1])
    new_expected.extend(expected[i_low : i_high+1])

    # only merge upper tail if there's something beyond the last "good" bin
    if i_high < M - 1:
        upper_count = counts[i_high+1 : ].sum()
        upper_exp   = expected[i_high+1 : ].sum()
        new_counts.append(upper_count)
        new_expected.append(upper_exp)

    return np.array(new_counts), np.array(new_expected)

def hellinger_distance(x1, x2):
    '''
    Compute the Hellinger distance between two probability distributions.
    This is a measure of the similarity between two probability distributions.
    '''
    h = 1/np.sqrt(2)*np.sqrt(np.sum((np.sqrt(x1) - np.sqrt(x2))**2))
    return h

def kolmogorov(M, N, k, draws):
    """
    Kolmogorov-Smirnov test for uniformity of empirical draws.
    M: max number of the draw
    N: number of draws
    k: position of the number in the draw
    draws: empirical draws (1-indexed)
    """
    # parameters for Beta(a=k, b=N+1-k)
    a, b = k, N+1-k
    # empirical draws
    u = Beta.cdf((draws-1)/(M-1), a, b)
    D, p_value = kstest(u, 'uniform')

    return D,p_value

def burstiness(gaps: np.ndarray) -> float:
    """
    Compute the burstiness of a point process given its inter-arrival 
    times array 'gaps' (e.g. days or draws between events).
    B = (CV - 1) / (CV + 1), can be negative for regular processes.
    """
    if gaps.size < 2:
        return 0.0
    mu = gaps.mean()
    sigma = gaps.std(ddof=1)
    cv = sigma/mu
    return (cv - 1)/(cv + 1)


def fano_from_counts(counts: np.ndarray) -> float:
    """
    Given an array of event counts in equal-length windows,
    compute the Fano factor = variance(counts) / mean(counts).
    """
    if counts.size < 2:
        raise ValueError("Need at least two windows to compute Fano")
    m = counts.mean()
    v = counts.var(ddof=1)
    if m == 0:
        return np.nan
    return v / m

def chi_analysis(df_data:pd.DataFrame, lotto_config:LottoConfigEntry, analysis_offset:int=0) -> pd.DataFrame:
    '''
    Chi-squared analysis of empirical draws.
    :param df_data: DataFrame containing the empirical draws.
    :param lotto_config: LottoConfigEntry containing the configuration for the lottery.
    :param analysis_offset: Number of draws to consider for the analysis (default is 0, meaning all draws).
    :return: DataFrame containing the chi-squared statistics, degrees of freedom, p-values, total variation distance, KL divergence, and merged KL divergence for each position.
    '''
    kl_pos = []
    tv_pos = []
    kl_merged_pos = []
    chi_pos = []
    chi_p = []
    df_results = pd.DataFrame(columns=['position', 'chi2_stat', 'df', 'p_value', 'tv_distance', 'kl_divergence', 'kl_merged'])
    for pos in range(len(lotto_config.cols)-1):
        col = lotto_config.cols[pos]
        n = lotto_config.maxval[pos]
        X = df_data[col]
        if analysis_offset > 0 :
            X = X[:analysis_offset]
        M = lotto_config.maxval[pos]
        counts = manual_histo_counts(X, M)
        T = len(X.values)
        pmf = order_stat_pmf(M=M, k=pos+1)                # length‐M array of theoretical probabilities
        pmf = (pmf + 1e-10)/pmf.sum()
        expected = T * pmf                        # expected count in each bin
        counts_m, expected_m = merge_tail_bins(counts, expected, min_exp=5)

        # degrees of freedom is now (# merged bins – 1)
        df_chi2 = len(counts_m) - 1

        chi2_stat = ((counts_m - expected_m)**2 / expected_m).sum()
        p_value   = 1 - chi2.cdf(chi2_stat, df_chi2)

        chi_p.append(p_value)
        print(f"D{pos+1}: χ² = {chi2_stat:.2f}, df = {df_chi2}, p = {p_value:.3f}")
        chi_pos.append(chi2_stat)
        # total‐variation distance
        tv = 0.5 * np.abs(counts/T - pmf).sum()
        print("Total variation:", tv)
        tv_pos.append(tv)
        # KL divergence
        # after you have counts (length M) and pmf (length M), both normalized:
        p_emp = counts / T     # empirical prob.
        p_mod = pmf            # theoretical prob.

        # only keep the terms where p_emp > 0
        mask = p_emp > 0

        kl = (p_emp[mask] * np.log(p_emp[mask] / p_mod[mask])).sum()
        print("KL divergence:", kl)
        kl_pos.append(kl)
        # counts_m, expected_m from merge_tail_bins(...)

        p_emp_m   = counts_m / counts_m.sum()
        p_model_m = expected_m / expected_m.sum()

        mask = p_emp_m > 0
        kl_m = (p_emp_m[mask] * np.log(p_emp_m[mask] / p_model_m[mask])).sum()
        kl_merged_pos.append(kl_m)
        print("KL divergence (merged bins):", kl_m)
        df_results.loc[pos] = pd.Series({
            'position': col,
            'chi2_stat': chi2_stat,
            'df': df_chi2,
            'p_value': p_value,
            'tv_distance': tv,
            'kl_divergence': kl,
            'kl_merged': kl_m
        })
    return df_results

def order_stat_pmf(M:int, k:int, N:int=5)-> np.ndarray:
    '''
    Compute the probability mass function (PMF) for the k-th order statistic in a draw of N numbers from a set of M numbers.
    :param M: Maximum number in the draw (e.g., 70 for Mega-Millions).
    :param k: Position of the number in the draw (1-indexed).
    :param N: Total numbers drawn (default is 5).
    :return: PMF for the k-th order statistic.
    '''
    x = np.arange(1, M+1)
    pmf = comb(x-1, k-1) * comb(M-x, N-k) / comb(M, N)
    return pmf


def calc_p(mu:float,std:float,a:float,b:float,n:float,sample=None) -> tuple:
    '''
    Calculate the z-scores and probabilities for a given range [a, b] based on a normal distribution with mean mu and standard deviation std.
    :param mu: Mean of the normal distribution.
    :param std: Standard deviation of the normal distribution.
    :param a: Lower bound of the range.
    :param b: Upper bound of the range.
    :param n: Maximum value in the range (used for normalization).
    :param sample: Optional sample value to calculate the likelihood.
    :return: A tuple containing the z-scores (z1, z2, zmin, zmax), the numerator and denominator of the probability, and the truncated normal likelihood for the sample.
    '''
    z1 = (a-mu)/std
    z2 = (b-mu)/std
    zmin = (1-mu)/std
    zmax = (n-mu)/std
    numerator = stats.norm.cdf(z1) - stats.norm.cdf(z2)
    denom = stats.norm.cdf(zmax) - stats.norm.cdf(zmin)
    likelihood=0.0
    if not sample is None :
        likelihood = stats.truncnorm.pdf(sample,zmin,zmax,loc=mu,scale=std)
    return z1, z2, zmin, zmax, numerator, denom,likelihood

def probability_for_draw(test_ticket,lotto_config,mustd) :
    ticket_likelihood_ = 1.0
    log_likelihood_ = 0.0
    ntickets = 1
    cp_ = 1.0
    dfprobs_ = pd.DataFrame(columns=['draw','mu', 'std','z1','z2','zmin','zmax','numerator','denominator','a','b','prob','cumprob','ntickets'])
    for i in range(0,len(lotto_config.cols)) :
        col = lotto_config.cols[i]
        n = lotto_config.maxval[i]
        mu,std = mustd[i]
        b = mu - std
        a = mu + std
        if b < 1 :
            b = 1
        if a > n :
            a = n
        ntickets *= int(abs(math.ceil(a) - math.ceil(b)))
        z1,z2,zmin,zmax,numerator,denom,likelihood = calc_p(mu,std,a,b,n,test_ticket[i])
        ticket_likelihood_ *= likelihood
        log_likelihood_ += -np.log(likelihood)
        p = numerator / denom
        cp_ *= p
        dfprobs_.loc[i] = pd.Series({'draw':col,'mu':mu,'std':std,'z1':z1,'z2':z2,'zmin':zmin,'zmax':zmax,'numerator':numerator,'denominator':denom,'a':a,'b':b,'prob':p,'cumprob':cp_,'ntickets':ntickets})
        px = 0.0
        if i < 5:
            px = comb_probability_position(n, 5, i+1, test_ticket[i])
        else :
            px = 1.0 / n
        print(f'Position {i} {col}, predicted {test_ticket[i]}, likelihood={likelihood}, px={px}')
    return dfprobs_, ticket_likelihood_, log_likelihood_, ntickets

# returns the joint probability of the ticket given the bounds of the positions
# for the ticket, and the negative log likelihood of that probability
#
def calc_bounded_p(ticket, lotto_config) :
    # ticket is [d1,d2,...,pb]
    # p = 1 / (x_(i+1) - x_(i-1) - 1)
    p = np.zeros((6,))
    for i in range(0,5) :
        low = 1
        if i > 0 :
            low = ticket[i-1]
        N = lotto_config.maxval[i]
        high = N
        if i < 4 :
            high = ticket[i+1]
        deno = high - low - 1
        #
        # When the range of values is zero, we are in a tight bound which
        # results in a single selection probability with 100% certainty. This
        # forces the denominator to 1 to give a p=1 probability. The log(1) is
        # zero so it would not contribute to the NLL, which is the real metric
        # we are using.
        if deno == 0 :
            deno = 1
        p[i] = 1.0/deno # the range of values for ticket i
    p[5] = 1.0/(lotto_config.maxval[5])
    P = np.prod(p)
    NLL = -np.log(p).sum()
    return P,NLL

def comb_probability_position(N, M, i, k):
    """
    Compute the probability that the i-th smallest number in an M-number draw from [1, N] is exactly k.
    From ChatGPT o3-mini-high
    The formula is:
        P(x_i = k) = (comb(k-1, i-1) * comb(N-k, M-i)) / comb(N, M)
    Parameters:
        N : int - total number of available numbers.
        M : int - total numbers drawn.
        i : int - draw position (1-indexed, i=1 is the smallest).
        k : int - the number for which to compute the probability.
    Returns:
        float - the probability that position i equals k.
    """
    # Valid k must satisfy: i <= k <= N - (M - i)
    if k < i or k > N - (M - i):
        return 0
    #print(f'i={i},k={k},M={M},N={N}')
    numerator = math.comb(k - 1, i - 1) * math.comb(N - k, M - i)
    denominator = math.comb(N, M)
    return numerator / denominator

def joint_likelihood(sequence, N, M):
    """
    Compute the joint likelihood (unconditional) for an entire sorted sequence.
    From ChatGPT o3-mini-high
    Parameters:
        sequence : list of int - the sorted draw numbers (length should equal M).
        N : int - total number of available numbers.
        M : int - total numbers drawn.
    Returns:
        float - the product of the individual probabilities.
    """
    L = 1.0
    for i, k in enumerate(sequence, start=1):
        p = comb_probability_position(N, M, i, k)
        L *= p
    return L

def nll_joint_likelihood(sequence, N, M):
    """
    Compute the negative log joint likelihood (unconditional) for an entire sorted sequence.
    From ChatGPT o3-mini-high
    Parameters:
        sequence : list of int - the sorted draw numbers (length should equal M).
        N : int - total number of available numbers.
        M : int - total numbers drawn.
    Returns:
        float - the product of the individual probabilities.
    """
    L = 0.0
    for i, k in enumerate(sequence, start=1):
        p = comb_probability_position(N, M, i, k)
        L += np.log(p)
    return -L

def bhattacharyya_distance(hist1, hist2, epsilon=1e-10):
    """
    Compute the Bhattacharyya distance between two probability distributions.
    From ChatGPT o3-mini-high

    Parameters:
        hist1 (array-like): First histogram (must sum to 1).
        hist2 (array-like): Second histogram (must sum to 1).
        epsilon (float): A small value to prevent log(0).

    Returns:
        float: The Bhattacharyya distance.
    """
    # Convert inputs to numpy arrays

    hist1 = np.array(hist1, dtype=float)
    hist2 = np.array(hist2, dtype=float)
    
    # Normalize the histograms in case they're not already normalized
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Compute the Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    # Clip the coefficient to avoid taking the log of 0
    bc = np.clip(bc, epsilon, 1.0)
    
    # Compute the Bhattacharyya distance
    distance = -np.log(bc)
    return distance

def calc_beta_p(k, N, M, a, b, sample=None):
    """
    Compute probabilities under the k-th order-statistic Beta(k, N+1-k) 
    on the discrete support {1,...,M}, treated as a continuous Beta on [1,M].
    
    Arguments:
      k       order-statistic index (1 ≤ k ≤ N)
      N       total number of draws per ticket (e.g. 5)
      M       maximum ball number (e.g. 70 for Mega-Millions)
      a, b    integer bounds of interest (inclusive)
      sample  single value or array of observed draws to get PDF at
      
    Returns:
      t_lower     scaled lower endpoint = (a-1)/(M-1)
      t_upper     scaled upper endpoint = (b-1)/(M-1)
      interval_p  Beta-CDF(t_upper) - Beta-CDF(t_lower)
      pdf_vals    if sample is not None, an array of Beta-PDF(sample_scaled)/(M-1);
                  otherwise None
    """
    # shape parameters
    alpha = k
    beta_param = N + 1 - k

    # scale a,b into [0,1]
    t_lower = (a - 1) / (M - 1)
    t_upper = (b - 1) / (M - 1)

    # probability mass over [a,b]
    interval_p = Beta.cdf(t_upper, alpha, beta_param) - Beta.cdf(t_lower, alpha, beta_param)

    pdf_vals = None
    if sample is not None:
        # allow scalar or array
        arr = np.atleast_1d(sample)
        t = (arr - 1) / (M - 1)
        # Beta.pdf is normalized on [0,1]; divide by (M-1) for the dx scale on [1,M]
        pdf_vals = Beta.pdf(t, alpha, beta_param) / (M - 1)
        # if input was scalar, return a scalar
        if np.isscalar(sample):
            pdf_vals = pdf_vals[0]

    return t_lower, t_upper, interval_p, pdf_vals

def sample_size_prop(p, delta, alpha=0.05):
    """
    Min T so that a binomial p̂ is within ±delta of p
    at (1-alpha) confidence.
    """
    z = norm.ppf(1 - alpha/2)
    return int(np.ceil(p*(1-p)*z**2 / delta**2))

def sample_size_ks(delta, alpha=0.05, power=0.8):
    """
    Approximate T needed for one-sample KS to detect shift Δ=delta
    with significance alpha and power (1-beta)=power.
    """
    K_alpha = np.sqrt(-0.5 * np.log(alpha/2))
    beta    = 1 - power
    K_beta  = np.sqrt(-0.5 * np.log(beta))
    return int(np.ceil((K_alpha + K_beta)**2 / delta**2))

def moe(p, n):
    Z = 1.96 # 95% confidence
    error = Z*np.sqrt(p*(1-p)/n)
    return error

# 1) Margin‐of‐error
def margin_error(p, T, alpha):
    z = norm.ppf(1 - alpha/2)
    return z * np.sqrt(p*(1-p)/T)

# 2) KS power approximation
def ks_power(delta, T, alpha):
    K_alpha = np.sqrt(-0.5*np.log(alpha/2))
    # find K_beta from sqrt(T)*delta - K_alpha
    K_beta  = np.sqrt(T)*delta - K_alpha
    beta = np.exp(-2 * K_beta**2)
    return 1 - beta


def compute_beta(df_data,lotto_config,analysis_offset=0):
    # no beta for the multiplier
    beta = np.zeros((lotto_config.maxval[0],len(lotto_config.cols)-1))
    x_ = np.zeros((len(lotto_config.cols)-1,4)) # [col,(mean|std|min|max)]
    for pos in range(0,len(lotto_config.cols)-1) :
        col = lotto_config.cols[pos]
        print(f'Computing Position {col} Probabilities')
        if not df_data is None :
            X = df_data[col]
        if analysis_offset > 0 :
            X = X[:analysis_offset]
        x_[pos,0] = X.mean()
        x_[pos,1] = X.std()
        x_[pos,2] = X.min()
        x_[pos,3] = X.max()
        for i in range(1, lotto_config.maxval[pos]+1) :
            beta[i-1,pos] = (i-1)**(pos+1-1)*(lotto_config.maxval[pos]-i)**(5-pos-1)
        beta[:,pos] /= beta[:,pos].sum()
    return beta,x_

def compute_pmf(df_data=None,np_data=None,lotto_config=None,analysis_offset=0):
    p_ = np.zeros((lotto_config.maxval[0],len(lotto_config.cols)))
    x_ = np.zeros((len(lotto_config.cols),5)) # [col,(mean|std|min|max|error)]
    for pos in range(0,len(lotto_config.cols)) :
        col = lotto_config.cols[pos]
        if not df_data is None :
            X = df_data[col]
        elif not np_data is None :
            X = np_data[:,pos]
        if analysis_offset > 0 :
            X = X[:analysis_offset]
        n = X.shape[0]
        x_[pos,0] = X.mean()
        x_[pos,1] = X.std()
        x_[pos,2] = X.min()
        x_[pos,3] = X.max()
        #
        # Compute the number of times each position occurs, making our own histogram manually
        #
        for i in range(1, lotto_config.maxval[pos]+1) :
            p_[i-1,pos] = np.where(X == i, 1, 0).sum()
            # print(f'... in Position {col}, {i} occurs {p_[i-1,pos]} times')
        p_[:,pos] /= X.shape[0]
        sx = p_[:,pos].sum()
        # Sanity check on the normalized probabilities, which might need a little fudge because of numerical drift
        if(sx < 0 or sx > (1.0 + 1e-8)) :
            print(f'Unexpected large cumulative prob {sx}')
        p_i = p_[:,pos].max() # assume this is the truth
        error = moe(p_i, n)
        x_[pos,4] = error
        print(f'Error is {error} or {error*100:.2f}% at 95% confidence assuming p_i = {p_i} and n = {n}')
        print(f'{col}: pmin={p_[:,pos].min()}, pmax={p_[:,pos].max()}, pstd={p_[:,pos].std()}, pmean={p_[:,pos].mean()}')
        print(f'{col}: min={x_[pos,2]}, max={x_[pos,3]}, std={x_[pos,1]}, mean={x_[pos,0]}')
    return p_,x_

def histogram_comparison(histo_ref : list, histo_emp : list, lotto_config : LottoConfigEntry) -> dict:
    """
    Compare histograms of reference and empirical data.
    Returns a dictionary with divergence metrics.
    """
    results = {}
    for pos in range(len(lotto_config.cols)):
        col = lotto_config.cols[pos]
        x1_counts = histo_ref[pos]
        x2_counts = histo_emp[pos]
        x1_norm = x1_counts / x1_counts.sum()
        x2_norm = x2_counts / x2_counts.sum()

        # Compute divergences
        results[col] = {
            'B': bhattacharyya_distance(x1_counts, x2_counts),
            'H': hellinger_distance(x1_norm, x2_norm),
            'JS': jensenshannon(x1_norm, x2_norm),
            'W': wasserstein_distance(x1_norm, x2_norm)
        }
    return results

def manual_histo_counts(data, max_value) :
    '''
    Compute a histogram for each of the draw values defined in the [1,max_value] range
    :param data: The data to compute the histogram from.
    :param max_value: The maximum value in the data range (e.g., 70 for Mega-Millions).
    :return: A histogram array where each index corresponds to the count of that value in the data.
    1-indexed, so index 0 corresponds to value 1.
    '''
    h = np.zeros(max_value)
    for i in range(0,max_value) :
        h[i] = np.where(data == i+1, 1, 0).sum()
    return h

def manual_histo(data, max_value, ax=None,orientation='vertical',color='blue',edgecolor='black',alpha=0.6) :
    '''
    Calls manual_histo_counts to compute the histogram and then plot it.
    :param data: The data to compute the histogram from.
    :param max_value: The maximum value in the data range (e.g., 70 for Mega-Millions).
    :param ax: Optional matplotlib Axes object to plot the histogram on.
    :param orientation: 'vertical' or 'horizontal' for the histogram orientation.
    :param color: Color of the histogram bars.
    :param edgecolor: Color of the edges of the histogram bars.
    :param alpha: Transparency level of the histogram bars.
    '''
    h = manual_histo_counts(data, max_value)
    if not ax is None :
        if orientation == 'vertical' :
            ax.bar(range(1,max_value+1),h, color=color, lw=.5,align='center',edgecolor=edgecolor,alpha=alpha)
        else :
            ax.barh(range(1,max_value+1),h, color=color, lw=.5,align='center',edgecolor=edgecolor,alpha=alpha)
    return h

def compare_to_order_stat_pmf(df:pd.DataFrame, lotto_config:LottoConfigEntry):
    kl_pos = []
    tv_pos = []
    kl_merged_pos = []
    chi_pos = []
    chi_p = []
    for pos in range(len(lotto_config.cols)-1):
        col = lotto_config.cols[pos]
        n = lotto_config.maxval[pos]
        X = df[col].to_numpy().astype(int)
        M = lotto_config.maxval[pos]
        counts = manual_histo_counts(X, M)
        T = X.shape[0]
        pmf = order_stat_pmf(M=M, k=pos+1)
        pmf = (pmf + 1e-10)/pmf.sum()
        expected = T * pmf                        # expected count in each bin
        counts_m, expected_m = merge_tail_bins(counts, expected, min_exp=5)

        # degrees of freedom is now (# merged bins – 1)
        df_chi2 = len(counts_m) - 1

        chi2_stat = ((counts_m - expected_m)**2 / expected_m).sum()
        p_value   = 1 - chi2.cdf(chi2_stat, df_chi2)

        chi_p.append(p_value)
        chi_pos.append(chi2_stat)
        # total‐variation distance
        tv = 0.5 * np.abs(counts/T - pmf).sum()
        tv_pos.append(tv)
        # KL divergence
        # after you have counts (length M) and pmf (length M), both normalized:
        p_emp = counts / T     # empirical prob.
        p_mod = pmf            # theoretical prob.

        # only keep the terms where p_emp > 0
        mask = p_emp > 0

        kl = (p_emp[mask] * np.log(p_emp[mask] / p_mod[mask])).sum()
        kl_pos.append(kl)
        # counts_m, expected_m from merge_tail_bins(...)

        p_emp_m   = counts_m / counts_m.sum()
        p_model_m = expected_m / expected_m.sum()

        mask = p_emp_m > 0
        kl_m = (p_emp_m[mask] * np.log(p_emp_m[mask] / p_model_m[mask])).sum()
        kl_merged_pos.append(kl_m)
    # compare the final position to a uniform random distro
    red_col = len(lotto_config.cols)-1
    col = lotto_config.cols[red_col]
    observed_series = df[col].value_counts().sort_index()
    max_red = lotto_config.maxval[-1]  # last column is the red ball
    all_indices = np.arange(1, max_red + 1)
    obs_counts = observed_series.reindex(all_indices, fill_value=0).astype(int).values
    total_draws = obs_counts.sum()

    if total_draws == 0:
        raise ValueError(f"No draws found in column '{red_col}'. Did you pass the correct column name?")

    # ──────────────── 3) Build the “expected” array under perfect uniformity ────────────────
    # Each of the max_red buckets would be expected to get (total_draws / max_red) counts.
    exp_count_per_number = total_draws / float(max_red)
    exp_counts = np.full(shape=obs_counts.shape, fill_value=exp_count_per_number, dtype=float)

    # ──────────────── 4) Run scipy.stats.chisquare to get χ² statistic & p‐value ────────────────
    chi2_stat, p_value = chisquare(f_obs=obs_counts, f_exp=exp_counts)

    # ──────────────── 5) Compute empirical pmf vs. uniform pmf → TV & KL divergences ────────────────
    p_obs = obs_counts / total_draws
    p_uniform = np.ones_like(p_obs) / max_red

    # Total‐variation distance = ½ * sum |p_obs − p_uniform|
    tv_dist = 0.5 * np.sum(np.abs(p_obs - p_uniform))

    # KL‐divergence = sum p_obs * log(p_obs / p_uniform), treating p_obs=0 as zero contribution
    kl_div = np.sum(np.where(p_obs > 0, p_obs * np.log(p_obs / p_uniform), 0.0))
    kl_pos.append(kl_div)
    kl_merged_pos.append(kl_div)
    tv_pos.append(tv_dist)
    chi_pos.append(chi2_stat)
    chi_p.append(p_value)
    return kl_pos, tv_pos, kl_merged_pos, chi_pos, chi_p

