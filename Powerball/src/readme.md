# Running The Lottery Statistical Analysis
----
You will need a python environment. Create the environment using the env.yaml provided.

> conda env create -f env.yaml

No specific version of python other than it being modern. The paper was created using python
version 3.12.9, pandas 2.2.3, and numpy 1.26.4.

There are several python scripts in this directory:

> jackpotanalysis.py : the main analysis script which creates all of the figures
>
> LottoConfig.py : meta data class for lottery game data
>
> lottostats.py : helpful functions for computing statistics on the data
>
> montecarlo.py : the monte carlo simulation code for simulating ticket draws
>
> PowerballEra.py : a class that encapsulates the data and processing for a single era

## Steps For Creating The Paper's Contents

Everything will be created in a subdirectory named *powerball*.

### Create the PKL files and the monte carlo simulation

> python jackpotanalysis.py --mc 1000 > mc_output.txt

This will create the PKL files, one for each era. Also created is the montecarlo_summary_stats.csv file and
other intermediate csv files used as reference. A key file created in this stage is the manifest file. 
That file is created in a subdirectory named *powerball*, e.g. powerball/jackpot_analysis.manifest. The manifest file only contains paths to the pkl files:

* powerball\powerball_era_20151004_present_objects.pkl
* powerball\powerball_era_19971101_20081112_objects.pkl
* powerball\powerball_era_20081112_20120115_objects.pkl
* powerball\powerball_era_20120115_20151004_objects.pkl

### Create the figures

> python jackpotanalysis.py --manifest powerball\jackpot_analysis.manifest > figures_out.txt

Running this command will create all of the figures used in the paper.
