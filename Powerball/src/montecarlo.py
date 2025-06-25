import sys
import random as random
import os
import numpy as np
import pandas as pd
import argparse
import LottoConfig as LottoConfig
from LottoConfig import LottoConfigEntry
import datetime as dt


def time_seed() -> None:
    """
    Generate a random seed based on the current time.
    This is useful for reproducibility in simulations.
    """
    return int(dt.datetime.now().timestamp() * 1000) % 15000000

def montecarlo_lottery(seed:int, tickets_to_run:int, lotto_config:LottoConfigEntry,runs:int=1) -> list[np.ndarray]:
    '''
    Run a Monte Carlo simulation for a lottery configuration.
    :param seed: Random seed for reproducibility, None for time-based seed
    :param tickets_to_run: Number of tickets to simulate
    :param lotto_config: Configuration for the lottery (LottoConfigEntry)
    :param runs: Number of simulation runs to perform
    :return: List of numpy arrays containing the simulated lottery tickets
    '''
    multiplier_col = len(lotto_config.cols)-1
    print(f'Running Monte Carlo lottery simulation with {tickets_to_run} tickets')
    if not seed is None :
        random.seed(seed)
    history = []
    if seed is None :
        random.seed(time_seed())
    else:
        random.seed(seed)
    for rn in range(0,runs) :
        u_data = np.zeros((tickets_to_run,len(lotto_config.cols)+1),dtype=int)
        for t in range(0,tickets_to_run) :
            r = []
            #
            # Each ticket gets a new set of balls to draw from
            #
            white_balls = [z for z in range(1,lotto_config.maxval[0]+1)]
            for j in range(0,multiplier_col):
                ival = random.randint(1,len(white_balls))
                r.append(white_balls[ival-1])
                white_balls.remove(white_balls[ival-1])
            r.sort()
            for j in range(0,multiplier_col):
                u_data[t,j] = r[j]
        # Run the multiplier on its own distribution
        if(seed is None) :
            random.seed(time_seed())
        for t in range(0,tickets_to_run) :
            u_data[t,multiplier_col] = random.randint(1,lotto_config.maxval[multiplier_col])
            u_data[t,multiplier_col+1] = rn + 1  # Store the run number
        history.append(u_data)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Monte Carlo lottery simulation.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--tickets', type=int, default=1000, help='Number of tickets to simulate')
    parser.add_argument('--lotto', type=str, default='superlotto', help='Lottery type (e.g., superlotto, powerball)')
    parser.add_argument('--runs', type=int, default=1, help='Number of simulation runs to perform')
    args = parser.parse_args()
    lotto = args.lotto.lower()
    if lotto not in LottoConfig().keys():
        print(f"Invalid lottery type: {lotto}. Valid types are: {LottoConfig().lotto_types()}")
        sys.exit(1)
    lotto_config = LottoConfig()[lotto]
    print(f"Running Monte Carlo simulation for {lotto} with {args.tickets} tickets and seed {args.seed}")
    u_data = montecarlo_lottery(args.seed, args.tickets, lotto_config, args.runs)
    df = pd.DataFrame(u_data, columns=lotto_config.cols+['Run'])
    str_seed = str(args.seed) if args.seed is not None else 'time_seed'
    df.to_csv(f'{lotto}-montecarlo-{args.tickets}-{str_seed}.csv', index=False)
    print(f"Results saved to {lotto}-montecarlo-{args.tickets}-{args.seed}.csv")
    print("Simulation complete.")