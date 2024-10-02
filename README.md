# IncDISCES

In complex event processing (CEP), queries are evaluated continuously over
streams of events to detect situations of interest, thereby
facilitating reactive applications.
However, users often lack insights into the precise event pattern that
characterizes the situation, which renders the
definition of the respective queries challenging.
Once a database of finite, historic
streams, each containing a materialization of the situation of interest, is
available, automated query discovery supports users in the
definition of the desired queries. It constructs the queries that match a
certain share of the given streams, as determined by a support
threshold. Yet, upon changes in the database or changes of the support
threshold, existing discovery algorithms need to construct the resulting
queries from scratch, neglecting the queries obtained in previous runs.

We aim to avoid the resulting inefficiencies by techniques
for incremental query discovery. We first provide a theoretical analysis of
the problem context, before presenting algorithmic solutions to cope with
changes in the stream database or the adopted support threshold.

Our experiments using real-world data demonstrate that our incremental query
discovery reduces the runtimes by up to 1000 times compared to a baseline solution.

The algorithms can be run with:
```
python scr/updates.py
```

A set of experiments using NASDAQ and Google Cluster datasets
can be run using:
```
python experiments/update_experiments.py
```
During the run of the experiments, the progress
can be followed and results can be seen in 'results/evaluation.csv'.
