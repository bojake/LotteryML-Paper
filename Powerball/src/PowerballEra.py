import numpy as np
import pandas as pd
import lottostats as ls
import datetime as dt
import LottoConfig as lc
import math
import montecarlo as mc
from dataclasses import dataclass
from typing import Tuple
from LottoConfig import LottoConfigEntry
from lottostats import burstiness

class PowerballEra(object):
    def __init__(self, title : str, config : LottoConfigEntry) -> None:
        self.version = 3
        self.T = 0
        self.title = title
        self.lotto = config.game
        self.config = config
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.era_string = f'{self.start_date:%Y-%m-%d} to '
        self.__save_file_name = f'{self.config.game}_era_{self.start_date:%Y%m%d}_'
        if self.end_date is None :
            self.era_string = self.era_string + 'Present'
            self.__save_file_name = self.__save_file_name + 'present.csv'
        else :
            self.era_string = self.era_string + f'{self.end_date:%Y-%m-%d}'
            self.__save_file_name = self.__save_file_name + f'{self.end_date:%Y%m%d}.csv'
        self.details = None
        self.data = None
        self.jackpots = None
        self.max_jackpot = None
        self.max_ticket = None
        self.pmf = None
        self.probs = None
        self.kl_by_position = None
        self.tv_by_position = None
        self.kl_merged_by_position = None
        self.chi_by_position = None
        self.chi_p_by_position = None
        self.mustd = None


    def save_filename(self, suffix=''):
        if suffix:
            return f'{self.__save_file_name[:-4]}_{suffix}.csv'
        else:
            return self.__save_file_name
    def __repr__(self):
        return f"PowerballEra(title={self.title}, start_date={self.start_date}, end_date={self.end_date})"
    
    def kl(self, pos):
        return self.kl_by_position[pos] if pos < len(self.kl_by_position) else None
    def tv(self, pos):
        return self.tv_by_position[pos] if pos < len(self.tv_by_position) else None
    def kl_merged(self, pos):
        return self.kl_merged_by_position[pos] if pos < len(self.kl_merged_by_position) else None
    def chi(self, pos):
        return self.chi_by_position[pos] if pos < len(self.chi_by_position) else None
    def chi_p(self, pos):
        return self.chi_p_by_position[pos] if pos < len(self.chi_p_by_position) else None
    
    def NLL(self):
        return self.details['NLLbeta'] if self.details is not None else None
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Monte Carlo simulation for the Powerball era
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _mc_simulate_draws(self, runs: int, seed:int = None,) -> list:
        """
        Simulate draws for the Powerball era.
        :param seed: Random seed for reproducibility
        :param runs: Number of runs to perform
        :return: list of simulated draws
        """
        return mc.montecarlo_lottery(seed, self.T, self.config, runs=runs)

    def _mc_compute_all_pmfs(self, simulated_draws: np.ndarray) -> np.ndarray:
        """
        Compute the PMF for all simulated draws.
        :param simulated_draws: 2D array of the simulated draws
        :return: PMF for each draw
        """
        self.mc_histo = []
        for pos in range(0,len(self.config.cols)) :
            h_mc = ls.manual_histo_counts(simulated_draws[:,pos], self.config.maxval[pos])
            self.mc_histo.append(h_mc)
        self.mc_histo_metrics = ls.histogram_comparison(self.mc_histo, self.histo_emp, self.config)

        M = self.config.maxval[0]
        runs = simulated_draws.shape[0] // self.T
        mc_all_pmf = np.zeros((runs,M,len(self.config.cols)))
        montecarlo_pmf = []
        k = 0
        r = 0
        while k < simulated_draws.shape[0] :
            u = simulated_draws[k:k+self.T,:]
            mc_probs,_ = ls.compute_pmf(np_data=u, lotto_config=self.config)
            montecarlo_pmf.append(mc_probs)
            mc_all_pmf[r,:] = mc_probs
            k += self.T
            r += 1
        return mc_all_pmf

    def _mc_pmf_envelope(self, pmfs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the lower and upper envelopes for the PMFs.
        :param pmfs: PMFs for each draw
        :return: Lower and upper envelopes
        """
        mc_lower = []
        mc_upper = []
        outliers = []
        for pos in range(0,len(self.config.cols)) :
            n = self.config.maxval[pos]
            m1 = np.percentile(pmfs[:,:,pos],  2.5, axis=0)
            m2 = np.percentile(pmfs[:,:,pos], 97.5, axis=0)
            outliers.append((self.pmf[0][:n,pos] < m1[:n]) | (self.pmf[0][:n,pos] > m2[:n]))
            mc_lower.append(m1)
            mc_upper.append(m2)
        return mc_lower, mc_upper, outliers

    def _mc_pmf_nll(self, runs:int = 1000) :
        """
        Compute the NLL for each simulate draw using the computed PMFs
        :param runs: Number of runs to perform
        :return: None
        0. For each run i
            a. Get the PMF for that run (shape (T,M))
            b. Get the simulated draws for that run (shape (T,))
            c. For each draw t in T
                i. Get the probability of the simulated draw from the PMF
                ii. Compute the NLL for that draw
            d. Store the NLLs for that run
        1. Compute the percentiles for the NLLs across all runs
        2. Store the results in self.mc_nll_upper, self.mc_nll_lower, self.mc_nll_median
        3. The shape of self.mc_all_nlls should be (runs, T)
        """
        nlls = []
        for i in range(runs) :
            pmf_i = self.mc_all_pmf[i,:,:]
            C = pmf_i.shape[1]
            cols = np.arange(C) 
            tickets_i = self.montecarlo_history[i] - 1
            tickets_i = tickets_i[:,:C] # remove the run number column
            # broadcast cols across the T dimension, so this returns (T, C)
            probs_i = pmf_i[tickets_i, cols]
            nll = np.sum(-np.log(probs_i),axis=1)
            nlls.append(nll)
        arr = np.array(nlls)
        if arr.shape != (runs, self.T) :
            print(f'Unexpected shape for MC NLLs: {arr.shape}, expected ({runs}, {self.T})')
            raise ValueError("Unexpected shape for MC NLLs")
        self.mc_all_nlls = arr # shape (runs, T)
        self.mc_nll_upper, self.mc_nll_lower = np.percentile(self.mc_all_nlls, [97.5, 2.5], axis=0)
        self.mc_nll_median = np.percentile(self.mc_all_nlls, 50, axis=0)

    def run_montecarlo(self, runs=1000, seed=None) :
        """
        Run a Monte Carlo simulation for the Powerball era.
        :param runs: Number of runs to perform
        :param seed: Random seed for reproducibility
        :return: DataFrame with the results of the simulation
        """
        print("$$$=================================================================================")
        print(f"Running {runs}x{self.T} Monte Carlo simulations for {self.era_string}")
        M = self.config.maxval[0]
        montecarlo_history = self._mc_simulate_draws(seed=seed, runs=runs)
        self.montecarlo_history = montecarlo_history
        #
        # run a histogram on the full history
        uniform_data = np.concatenate(montecarlo_history, axis=0)
        self.mc_all_pmf = self._mc_compute_all_pmfs(uniform_data)
        #
        # analyze the percentiles for comparison with the empirical
        #
        mc_lower, mc_upper, outliers = self._mc_pmf_envelope(self.mc_all_pmf)
        self.mc_lower = mc_lower
        self.mc_upper = mc_upper
        self.mc_outliers = outliers
        #
        # compute the NLL for each simulated draw using the computed PMFs
        #
        self._mc_pmf_nll(runs=runs)
        print(f"Completed {runs} Monte Carlo simulations for {self.era_string}")

    def compute_running_burstiness(self,details, window_days=90):
        # details: DataFrame with at least ['Date','is_jackpot']
        details = details.copy()
        details['Date'] = pd.to_datetime(details['Date'])
        
        # get jackpot dates only
        jack_dates = details.loc[details['is_jackpot']==1, 'Date'].sort_values()
        
        # prepare an array to hold B_t
        B_vals = np.full(len(details), np.nan)
        
        # for efficiency, convert jack_dates to numpy
        jack_np = jack_dates.to_numpy()
        
        for i, curr in enumerate(details['Date']):
            # define window [curr - W, curr]
            start = curr - pd.Timedelta(days=window_days)
            # select jackpots in that window
            mask = (jack_np >= start) & (jack_np <= curr)
            window_jacks = jack_np[mask]
            if len(window_jacks) > 1:
                gaps = np.diff(window_jacks).astype('timedelta64[D]').astype(int)
                B_vals[i] = burstiness(gaps)
            else:
                B_vals[i] = np.nan  # too few points
            
        details['running_burstiness'] = B_vals
        details['running_burstiness'] = details['running_burstiness'].interpolate()
        return details

    def analyze(self, data) :
        print(f'Analyzing {self.era_string} with {data.shape[0]} draws')
        kl_pos, tv_pos, kl_merged_pos, chi_pos, chi_p = ls.compare_to_order_stat_pmf(data, self.config)
        self.kl_by_position = kl_pos
        self.tv_by_position = tv_pos
        self.kl_merged_by_position = kl_merged_pos
        self.chi_by_position = chi_pos
        self.chi_p_by_position = chi_p
        self.data = data
        self.details = None # pd.DataFrame(columns=lotto_config.cols+['Prob','NLL','Offset', 'jProb', 'jNLL', 'BoundedP', 'BoundedNLL','Winners','Jackpot','NLLbeta'] + [f'p{x}' for x in lotto_config.cols])
        self.mustd = []
        self.annotations = []
        self.histo_emp = []
        lotto_config = self.config
        min_T = ls.sample_size_prop(p=0.03, delta=0.01)
        print(f'Minimum sample size for 3% accuracy at 1% confidence is {min_T}, actual T={data.shape[0]}')
        min_T = 2 # override for now
        self.mimimum_sample_size = min_T
        print(f'Overriding minimum sample size to {min_T} for now to get a long history of past draws')
        for pos in range(0,len(lotto_config.cols)) :
            col = lotto_config.cols[pos]
            n = lotto_config.maxval[pos]
            X = data[col].to_numpy().astype(int)
            hh = ls.manual_histo(X, n, ax=None)
            self.histo_emp.append(hh)
        #
        # Run through all of the draws, analyzing the PMF and other statistics for the ticket. Stop
        # when the historical data count is too small for reliable PMF calculation.
        #
        p,x = ls.compute_pmf(df_data=data, lotto_config=lotto_config)
        self.pmf = (p,x)
        for k in range(1, self.data.shape[0]+1) :
            sample = None
            tmp_data = None
            sample = self.data.iloc[k-1,:]
            tmp_data = self.data.iloc[k:,:]
            if tmp_data.shape[0] == 0 :
                print(f'No data for {k} days from last draw, ending')
                return
            if tmp_data.shape[0] < min_T :
                print(f'Insufficient data for reliable PMF at {k} draws from last draw, need {min_T}, have {tmp_data.shape[0]}')
                break
            if self.details is None :
                self.details = pd.DataFrame(columns=lotto_config.cols+['Prob','NLL','Offset', 'jProb', 'jNLL', 'BoundedP', 'BoundedNLL','Winners','Jackpot','NLLbeta','Date','is_jackpot','inter_arrival'] + [f'p{x}' for x in lotto_config.cols])
            sample5 = sample[lotto_config.cols[:-1]].to_numpy().astype(int)
            p,x = self.pmf
            cumulative_p = []
            cp = 1.0
            self.probs = pd.DataFrame(columns=['draw','mu', 'std','z1','z2','zmin','zmax','numerator','denominator','a','b','prob','cumprob','KL','TV','KL-merged','Chi','ChiP'])
            sample_likelihood = 1.0
            log_likelihood = 0.0
            log_likelihood_beta = 0.0
            joint_ll = 0.0
            output_data = {}
            for i in range(0,len(lotto_config.cols)) :
                col = lotto_config.cols[i]
                sample_draw = None
                sample_draw = sample.iloc[i]
                n = lotto_config.maxval[i]
                output_data[col] = sample_draw
                mu,std,_,_,_ = self.pmf[1][i] #mustd[i]
                b = mu - std
                a = mu + std
                if b < 1 :
                    b = 1
                if a > n :
                    a = n
                z1,z2,zmin,zmax,numerator,denom,likelihood = ls.calc_p(mu,std,a,b,n,sample_draw)
                sample_likelihood *= likelihood
                log_likelihood += -np.log(likelihood)
                p = numerator / denom
                cp *= p
                cumulative_p.append(cp)
                output_data[f'p{col}'] = likelihood
                #print(f'Position {i} {col}, predicted {samples[i]}, likelihood={likelihood}')

                # do the beta distribution log likelihood
                N = len(lotto_config.cols)-1
                if i < N :
                    _, _, _, pdfs = ls.calc_beta_p(k=i+1, N=N, M=n, a=0, b=0, sample=sample_draw)
                    log_likelihood_beta += np.sum(-np.log(pdfs))
                else :
                    # mega is always 1 in max 
                    log_likelihood_beta += -np.log(1.0/n)
                self.probs.loc[i] = pd.Series({'draw':col,'mu':mu,'std':std,'z1':z1,'z2':z2,'zmin':zmin,'zmax':zmax,'numerator':numerator,'denominator':denom,'a':a,'b':b,'prob':p,'cumprob':cp,'KL':self.kl_by_position[i],'TV':self.tv_by_position[i],'KL-merged':self.kl_merged_by_position[i],'Chi':self.chi_by_position[i],'ChiP':self.chi_p_by_position[i]})

            M = len(lotto_config.cols)-1
            N = lotto_config.maxval[0]
            # sample.iloc[:M]
            jprob = ls.joint_likelihood(sample5,N=N,M=M)
            jnll = ls.nll_joint_likelihood(sample5,N=N,M=M)
            # print(dfprobs)
            # print(f'Sequence likelihood={sample_likelihood}, log-likelihood={log_likelihood}, beta likelihood={log_likelihood_beta}')
            output_data['Prob'] = sample_likelihood
            output_data['NLL'] = log_likelihood
            output_data['NLLbeta'] = log_likelihood_beta
            output_data['jNLL'] = jnll
            output_data['Date'] = sample['Date']
            output_data['Offset'] = k
            output_data['jProb'] = jprob
            boundedp,boundednll = ls.calc_bounded_p([sample.iloc[0], sample.iloc[1], sample.iloc[2], sample.iloc[3], sample.iloc[4], sample.iloc[5]],lotto_config)
            output_data['BoundedP'] = boundedp
            output_data['BoundedNLL'] = boundednll
            ratio = log_likelihood / jnll
            output_data['Winners'] = data.iloc[k-1]['Winners']
            output_data['Jackpot'] = data.iloc[k-1]['Jackpot']
            output_data['inter_arrival'] = 0
            output_data['is_jackpot'] = 1 if (data.iloc[k-1]['Jackpot'] == 1 or data.iloc[k-1]['Jackpot'] == 'Yes') else 0
            if data.iloc[k-1]['Jackpot'] == 1 or data.iloc[k-1]['Jackpot'] == 'Yes' :
                self.annotations.append((k,log_likelihood,np.log10(data.iloc[k-1]['Winners']), ratio,jnll,log_likelihood_beta))
            self.details.loc[k] = pd.Series(output_data)
            self.winners_log10 = np.log10(self.details['Winners'].to_numpy().astype(np.float64))
            # print(f'Analyzed {k} of {self.data.shape[0]} draws, details.rows={self.details.shape[0]}')
        # save the T, which is the number of tickets
        self.T = self.details.shape[0]
        # convert the date column to a datetime for proper date handling
        self.details['Date'] = pd.to_datetime(self.details['Date'])
        # Make a copy of the jackpot details
        self.jackpots = self.details[self.details['Jackpot'] == 'Yes'].copy()
        self.max_jackpot = self.jackpots[self.jackpots.NLLbeta==self.jackpots.NLLbeta.max()][self.config.cols+['NLLbeta','Winners','Date']]
        self.max_ticket = self.details[self.details.NLLbeta==self.details.NLLbeta.max()][self.config.cols+['NLLbeta','Winners','Date']]
        # make the inter arrival times for the burstiness calculation
        # after self.jackpots is built (it has a 'Date' column!)
        # preserve the inter_arrival gaps for graphing
        mask = self.details['is_jackpot'] == 1
        jackpot_dates = self.details.loc[mask, 'Date']
        jackpot_k = self.details.loc[mask, 'Offset']
        day_gaps = jackpot_dates.diff().dt.days.fillna(0).astype(int)
        day_gaps = jackpot_k.diff().fillna(0).astype(int)
        self.details['inter_arrival'] = 0
        self.details.loc[mask, 'inter_arrival'] = day_gaps

        # 1) Compute burstiness on actual day-gaps:
        dates = pd.to_datetime(self.jackpots['Date']).sort_values()
        day_gaps = dates.diff().dt.days.interpolate().to_numpy()
        day_gaps = self.jackpots['Offset'].diff().fillna(0).to_numpy().astype(int)
        self.jackpot_burstiness = ls.burstiness(day_gaps)

        # 2) Compute Fano factor on counts per fixed‐length window (e.g. 90 days)
        window_days = 90
        start, end = dates.min(), dates.max()
        counts = []
        curr = start
        while curr < end:
            nxt = curr + pd.Timedelta(days=window_days)
            counts.append(((dates >= curr) & (dates < nxt)).sum())
            curr = nxt
        counts = np.array(counts)
        self.jackpot_fano = ls.fano_from_counts(counts)
        rows = self.details.shape[0]
        print(f'~~ Before burstiness, details has shape {self.details.shape}')
        self.details = self.compute_running_burstiness(self.details)
        print(f'~~ After burstiness, details has shape {self.details.shape}')
        if rows != self.details.shape[0] :
            print(f'Warning: details row count changed from {rows} to {self.details.shape[0]} after burstiness computation')
            raise ValueError("Details row count changed after burstiness computation")

    def report(self):
        print("~~~~~~~~~~ Powerball Era Report ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f'Powerball Era: {self.era_string}')
        for y1,y2,y3,y4,y5,y6 in self.annotations:
            jpdate = self.details.iloc[y1]['Date']
            print(f'Jackpot {jpdate} {y1} - NLL={y2}, log winners = {y3}, ratio={y4}, jNLL={y5}, NLL beta={y6}')
        print(self.details.describe())
        print(self.details.head())
        print(self.details.tail())
        print("Max NLL ticket")
        print(self.max_ticket)
        print("Max NLL jackpot record")
        print(self.max_jackpot)
        sorted = self.details.sort_values(by='NLLbeta', ascending=True)
        print(sorted.head(5))
        print(sorted.tail(5))
        print(self.details['Winners'])
        print(f'Winners log10 max={self.winners_log10.max()}, min={self.winners_log10.min()}')
        print(f'Winners log10 mean={self.winners_log10.mean()}, std={self.winners_log10.std()}')
        print(f'Max NLLbeta log10 winners={self.winners_log10[self.details.NLLbeta==self.details.NLLbeta.max()]}')

