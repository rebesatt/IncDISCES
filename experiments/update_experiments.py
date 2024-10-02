
import sys
# sys.path.append('.')
from collections import deque
import time
import os
import pandas as pd
import gzip
import collections
from copy import deepcopy
from statistics import mean
import seaborn as sns

sys.path.append('.')
from src.discovery_bu_multidim import _next_queries_multidim, domain_unified_discovery_smarter
from src.sample_multidim import MultidimSample
from src.query_multidim import MultidimQuery
from src.updates import update_query_set


def main():
    result_path =f'experiments/results'
    file_name = f'{result_path}/evaluation'
    file_path = f'{file_name}.csv'
    
    results = []
    if not os.path.isfile(file_path):
        repetition = 5
        run = 0
        for abstraction in ['F1', 'F2', 'F3', 'G1', 'G2', 'G3']:
            
            if abstraction[0] == 'F':
                mod = 'finance'
            else:
                mod = 'google'
            current_run = f'{mod}_{abstraction}'
            if os.path.isfile(f'{result_path}/{current_run}.csv'):
                current_df = pd.read_csv(f'{result_path}/{current_run}.csv',
                                         header=0, index_col=0)
                results.extend(current_df.values)
                dataframe = pd.DataFrame(results, 
                                    columns=current_df.columns) 
            else:
                if abstraction == 'F1':
                    sample_path = 'datasets/finance_query1.txt.gz'
                    trace_length = 50
                    sample_size = 79
                    max_query_length = -1
                    supp1 = .95
                    supp2 = .95


                elif abstraction == 'F2':
                    trace_length = 40
                    sample_size = 1000
                    max_query_length = -1
                    sample_path = 'datasets/finance_query2.txt.gz'
                    supp1 = .97
                    supp2 = .97

                elif abstraction == 'F3':
                    trace_length = 25
                    sample_size = 250
                    max_query_length = -1
                    sample_path = 'datasets/finance_query3.txt.gz'
                    supp1 = .95
                    supp2 = .95

                elif abstraction == 'G1':
                    trace_length = 7
                    sample_size = 1000
                    max_query_length = -1
                    sample_path = 'datasets/google_query1_status1.txt.gz'
                    supp1 = 1
                    supp2 = 1

                elif abstraction == 'G2':
                    trace_length = 7
                    sample_size = 1000
                    max_query_length = 4
                    sample_path = 'datasets/google_query2_status1.txt.gz'
                    supp1 = 1
                    supp2 = 1

                elif abstraction == 'G3':
                    trace_length = 7
                    max_query_length = 4
                    sample_size = 1000
                    sample_path = 'datasets/google_query3_status1.txt.gz'
                    supp1 = 1
                    supp2 = 1

                sample_list = []

                counter = 0
                file = gzip.open(sample_path, 'rb')
                for trace1 in file:
                    if counter == sample_size:
                        break

                    if trace_length == -1:
                        sample_list.append(' '.join(trace1.decode().split()))
                    else:
                        trace = ' '.join(trace1.decode().split()[-trace_length:])
                        sample_list.append(trace)

                    counter += 1
                file.close()
                j = abstraction
                samples2_list = []
                samples2_list.append(sample_list)
                for i in range (1,11):
                    samples2_list.append(sample_list[:-i])
                
                samples1 = [samples2_list[-1], samples2_list[0], samples2_list[0], samples2_list[0]]
                samples2 = [samples2_list[:-1], samples2_list[1:], [samples2_list[0]], [samples2_list[0]]]
                supports1 = [supp1, supp1, 0.75, 1]
                supports2 = [[supp2], [supp2], [0.8, 0.85, 0.9, 0.95, 1], [0.75, 0.8, 0.85, 0.9, 0.95]]
                up_mode_list = ['gen', 'spec', 'gen', 'spec']
                supp_sample = ['sample', 'sample', 'supp', 'supp']

                for samp1, samp2, sp1, sp2, up_mode, spsamp in zip(samples1, samples2, supports1, 
                                                                supports2, up_mode_list, supp_sample):
                    
                    file_path_result = f'{result_path}/{current_run}_{up_mode}_{spsamp}.csv'

                    if os.path.isfile(file_path_result):
                        current_df = pd.read_csv(file_path_result,
                                        header=0, index_col=0)
                        results.extend(current_df.values)
                        columns = current_df.columns
                        dataframe = pd.DataFrame(results,
                                    columns=columns)
                    else:
                        for _ in range(repetition):
                            results, columns = match_algos(samp1, samp2, 
                                                        sp1, sp2, results, up_mode, mod,
                                                        j, file_path, max_query_length=max_query_length)
                        dataframe = pd.DataFrame(results, columns=columns)
                        if len(samp2) !=1:

                            current_df = dataframe.loc[(dataframe['mode'] == abstraction) &
                                                    (dataframe['iteration'] == up_mode) &
                                                    (dataframe['support 1']== dataframe['support 2'] )]
                        else:
                            current_df = dataframe.loc[(dataframe['mode'] == abstraction) &
                                                    (dataframe['iteration'] == up_mode) &
                                                    (dataframe['support 1']!= dataframe['support 2'] )]
                        current_df.to_csv(file_path_result)

                    run += 1
                dataframe = pd.DataFrame(results, columns=columns)
                current_df = dataframe.loc[(dataframe['mode'] == abstraction)]
                current_df.to_csv(f'{result_path}/{current_run}.csv')
                
        if os.path.isfile(file_path):
            dataframe = pd.read_csv(file_path)
        else:
            columns = current_df.columns.values[1:]
            dataframe = pd.DataFrame(results, columns=columns)
            dataframe.to_csv(file_path)
    else:
        dataframe = pd.read_csv(file_path)
    create_plots(dataframe)
    return dataframe


def match_algos(sample1_list: list, samples2_list: list, supp1:float, supps2:list,
                results: list, iterations: int, j: int|str, mod: str,file_path: str, 
                discovery: list|None = None, max_query_length: int = - 1,
                only_types: bool = False) -> list:
    if not discovery:
        discovery = ['duc', 'updated']
    
    sample = MultidimSample()
    sample.set_sample(sample1_list)
    sample_size = len(sample1_list)
    result_dictionary = domain_unified_discovery_smarter(sample, supp1, max_query_length=max_query_length)
    orig_matching_dict_len = len(result_dictionary['matching_dict'])
    
    for upd_list in samples2_list:
        for supp2 in supps2:
            sample2 = MultidimSample()
            sample2.set_sample(upd_list)
            sample2_size = len(upd_list)
            trace_sample = MultidimSample()
            trace_sample.set_sample([sample1_list[0]])

            trace_length = sample1_list[0].count(' ') + 1
            sample.get_vertical_sequence_database()
            len_pattern = []
            patterns = []
            trace_pattern = 0
            pattern_cnt = 0
            avg_query_length = 0
            result_dict = {}
            
            att_vsdb = sample.get_att_vertical_sequence_database()
            vsdb = {}
            domain_cnt = sample._sample_event_dimension
            alphabet = set()
            gen_event= ';' * domain_cnt
            gen_event_list = [i for i in gen_event]
            patternset ={}
            for domain, dom_vsdb in att_vsdb.items():
                patternset[domain] = set()
                for key, value in dom_vsdb.items():
                    new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
                    vsdb[new_key] = value
                    if not only_types:
                        for item in value.keys():
                            if len(value[item]) >= 2:
                                patternset[domain].add(key)
                                break

            pattern_list = [0]*sample_size
            for pos_dict in vsdb.values():
                for trace, positions in pos_dict.items():
                    if len(positions) >= 2:
                        pattern_list[trace] = pattern_list[trace] + 1

            sample_sized_support = sample_size
            alphabet = {symbol for symbol,value in vsdb.items() if len(value) >= sample_sized_support}
            dim_sample_dict = sample.get_dim_sample_dict()
            sum_pattern_list = []
            sum_type_list = []
            for idx, _ in enumerate(sample._sample):
                sum_pattern_dom_list = []
                sum_type_dom_list = []
                for dom_sample in dim_sample_dict.values():
                    supported_alphabet = dom_sample.get_sample_supported_typeset()
                    trace_list = dom_sample._sample[idx].split()
                    trace_length = len(trace_list)
                    event_counter = collections.Counter(trace_list)
                    freq_dom_list = event_counter.most_common()
                    pattern_sum  = sum(tup[1] for tup in freq_dom_list if tup[1] > 1)
                    sum_pattern_dom_list.append(pattern_sum)
                    type_sum = sum(tup[1] for tup in freq_dom_list if tup[0].replace(';','') in supported_alphabet)
                    sum_type_dom_list.append(type_sum)
                domain_pattern_sums = sum(sum_pattern_dom_list)
                sum_pattern_list.append(domain_pattern_sums)
                domain_type_sums = sum(sum_type_dom_list)
                sum_type_list.append(domain_type_sums)

                for domain in att_vsdb.keys():
                    for _, value in att_vsdb[domain].items():
                        if list(value.keys())[0] == idx:
                            trace_pattern = max(trace_pattern, len(value[idx]))
                            if len(value[idx]) >= 2:
                                pattern_cnt +=1
                len_pattern.append(trace_pattern)
                patterns.append(pattern_cnt)
                trace_pattern = 0
                pattern_cnt = 0
            
            max_sum_pattern = min(sum_pattern_list, default=0)
            max_sum_type = max(sum_type_list, default=0)

            for matching in discovery:
                if matching == 'duc':
                    copy_sample = deepcopy(sample2)
                    start= time.time()
                    result_dict = domain_unified_discovery_smarter(copy_sample, supp2, max_query_length)
                    result= time.time()-start
                    
                    if result_dict:
                        queryset1 = result_dict['queryset']
                        queryset = queryset1
                        querycount = result_dict['querycount']
                        matchingset = set(result_dict['matching_dict'].keys())
                        searchspace = len(result_dict['matching_dict'])
                        query_lengths = 0
                        for query in queryset:
                            query_lengths += len(query.split())
                        if len(queryset) > 0:
                            avg_query_length = query_lengths / len(queryset)
                elif matching == 'updated':
                    orig_sample = deepcopy(sample)
                    copy_sample = deepcopy(sample2)
                    result_dict = deepcopy(result_dictionary)
                    start= time.time()
                    result_dict = update_query_set(result_dict, orig_sample, copy_sample, supp1, supp2, max_query_length=max_query_length)
                    result= time.time()-start
                    queryset2 = result_dict['queryset']
                    queryset = queryset2

                    searchspace = len(set(result_dict['matching_dict'].keys()))
                    assert matchingset == set(result_dict['matching_dict'].keys())
                if result_dict:
                    results.append([str(sample_size), str(sample2_size),str(supp1),str(supp2), str(trace_length),str(iterations), matching,str(result),
                                str(j), str(orig_matching_dict_len), queryset, mod, searchspace, str(max_sum_type),
                                str(max_sum_pattern),
                                str(max_query_length), str(len(alphabet)), str(mean(pattern_list)), str(avg_query_length)])
                columns = ['sample size 1', 'sample size 2', 'support 1', 'support 2', 'trace length','iteration', 'algorithm', 'time',
                        'iterations', 'queryset size', 'queryset', 'mode', 'searchspace', 'type sum',
                        'pattern sum', 'max query length',
                        'supported types', 'pattern types', 'avg query length']
                dataframe = pd.DataFrame(results, columns=columns)
                dataframe.to_csv(file_path)
    return results, columns

def create_plots(dataframe:pd.DataFrame):
    df_gen_sample = dataframe.loc[(dataframe['iteration'] == 'gen') &
                                 (dataframe['support 1']== dataframe['support 2'] )]
    df_spec_sample = dataframe.loc[(dataframe['iteration'] == 'spec') &
                                    (dataframe['support 1']== dataframe['support 2'] )]
    
    df_gen_supp = dataframe.loc[(dataframe['iteration'] == 'gen') &
                                (dataframe['support 1']!= dataframe['support 2'] )]
    
    df_spec_supp = dataframe.loc[(dataframe['iteration'] == 'spec') &
                                 (dataframe['support 1']!= dataframe['support 2'] )]
    
    results = []
    all_results = []
    sns.set_context('poster', font_scale=1.2) # rc={"lines.linewidth": 2.5})
    for mode in df_gen_sample['mode'].unique():
        orig_sample_size = df_gen_sample.loc[(df_gen_sample['mode']==mode)]['sample size 1'].min()
        for samp2_size in df_gen_sample['sample size 2'].unique():
            q1 = df_gen_sample.loc[(df_gen_sample['sample size 2']==samp2_size) & 
                                   (df_gen_sample['mode']== mode)]['queryset size'].min()
            q2 = df_gen_sample.loc[(df_gen_sample['sample size 2']==samp2_size) & 
                                   (df_gen_sample['mode']== mode)]['searchspace'].min()
            df_gen_sample_duc_time = df_gen_sample.loc[(df_gen_sample['sample size 2']==samp2_size) & 
                                                       (df_gen_sample['mode']== mode) &
                                                       (df_gen_sample['algorithm'] == 'duc' )]['time'].values
            df_gen_sample_upd_time = df_gen_sample.loc[(df_gen_sample['sample size 2']==samp2_size) & 
                                                       (df_gen_sample['mode']== mode) &
                                                       (df_gen_sample['algorithm'] == 'updated' )]['time'].values

            rel_timechange = [y / x for x,y in zip(sorted(df_gen_sample_upd_time), 
                                                   sorted(df_gen_sample_duc_time))]
            if mode[0]=='F':
                abstraction = 'finance'
            else:
                abstraction = 'google'
            for rtime in rel_timechange:
                results.append([samp2_size-orig_sample_size, mode, rtime, abstraction])
                all_results.append([q1-q2, mode, rtime,abstraction])
    columns = ['Added streams', 'Abstraction', 'Rel timechange', 'Dataset' ]
    df_plot = pd.DataFrame(results, columns=columns)
    grid = sns.relplot(data=df_plot, x='Added streams', y='Rel timechange', hue='Abstraction',
                       col='Dataset', kind='line', style='Abstraction', markers=True, legend=False)
    grid.set(yscale='log', yticks= [1,10,100], xticks=[5,10])
    grid.savefig('experiments/BTW2025/plots/gen_sample.pdf')

    results = []
    for mode in df_gen_supp['mode'].unique():
        
        for supp2 in df_gen_supp['support 2'].unique():
            q1 = df_gen_supp.loc[(df_gen_supp['mode']==mode)&
                                 ((df_gen_supp['support 2']==supp2))]['queryset size'].min()
            q2 = df_gen_supp.loc[(df_gen_supp['mode']==mode)&
                                 ((df_gen_supp['support 2']==supp2))]['searchspace'].min()
            df_gen_supp_duc_time = df_gen_supp.loc[(df_gen_supp['support 2']==supp2) & 
                                                       (df_gen_supp['mode']== mode) &
                                                       (df_gen_supp['algorithm'] == 'duc' )]['time'].values
            df_gen_supp_upd_time = df_gen_supp.loc[(df_gen_supp['support 2']==supp2) & 
                                                       (df_gen_supp['mode']== mode) &
                                                       (df_gen_supp['algorithm'] == 'updated' )]['time'].values

            rel_timechange = [y / x for x,y in zip(sorted(df_gen_supp_upd_time), 
                                                   sorted(df_gen_supp_duc_time))]
            if mode[0]=='F':
                abstraction = 'finance'
            else:
                abstraction = 'google'
            for rtime in rel_timechange:
                results.append([supp2, mode, rtime, abstraction])
                all_results.append([q1-q2, mode, rtime,abstraction])
    columns = ['New support', 'Abstraction', 'Rel timechange', 'Dataset' ]
    df_plot = pd.DataFrame(results, columns=columns)
    grid = sns.relplot(data=df_plot, x='New support', y='Rel timechange', hue='Abstraction',
                       col='Dataset', kind='line', style='Abstraction', markers=True)
    grid.set(yscale='log', yticks= [1,10,100], xticks=[0.8, 0.9, 1.0])
    grid.savefig('experiments/BTW2025/plots/gen_supp.pdf')

    results = []
    for mode in df_spec_sample['mode'].unique():
        orig_sample_size = df_spec_sample.loc[(df_spec_sample['mode']==mode)]['sample size 1'].min()
        for samp2_size in df_spec_sample['sample size 2'].unique():
            q1 = df_spec_sample.loc[(df_spec_sample['sample size 2']==samp2_size) & 
                                    (df_spec_sample['mode']== mode)]['queryset size'].min()
            q2 = df_spec_sample.loc[(df_spec_sample['sample size 2']==samp2_size) & 
                                    (df_spec_sample['mode']== mode)]['searchspace'].min()
            df_spec_sample_duc_time = df_spec_sample.loc[(df_spec_sample['sample size 2']==samp2_size) & 
                                                       (df_spec_sample['mode']== mode) &
                                                       (df_spec_sample['algorithm'] == 'duc' )]['time'].values
            df_spec_sample_upd_time = df_spec_sample.loc[(df_spec_sample['sample size 2']==samp2_size) & 
                                                       (df_spec_sample['mode']== mode) &
                                                       (df_spec_sample['algorithm'] == 'updated' )]['time'].values

            rel_timechange = [y / x for x,y in zip(sorted(df_spec_sample_upd_time), 
                                                   sorted(df_spec_sample_duc_time))]
            if mode[0]=='F':
                abstraction = 'finance'
            else:
                abstraction = 'google'
            for rtime in rel_timechange:
                results.append([orig_sample_size-samp2_size, mode, rtime, abstraction])
                all_results.append([q1-q2, mode, rtime,abstraction])
    columns = ['# Deleted streams', 'Abstraction', 'Rel timechange', 'Dataset' ]
    df_plot = pd.DataFrame(results, columns=columns)
    grid = sns.relplot(data=df_plot, x='# Deleted streams', y='Rel timechange', hue='Abstraction',
                       col='Dataset', kind='line', style='Abstraction', markers=True, legend=False)
    grid.set(yscale='log', yticks= [1,10,100], xticks=[5,10])
    grid.savefig('experiments/BTW2025/plots/spec_sample.pdf')

    results = []
    for mode in df_spec_supp['mode'].unique():
        for supp2 in df_spec_supp['support 2'].unique():
            q1 = df_spec_supp.loc[(df_spec_supp['support 2']==supp2) & 
                                (df_spec_supp['mode']== mode)]['queryset size'].min()
            q2 = df_spec_supp.loc[(df_spec_supp['support 2']==supp2) &
                                  (df_spec_supp['mode']== mode)]['searchspace'].min()
            df_spec_supp_duc_time = df_spec_supp.loc[(df_spec_supp['support 2']==supp2) & 
                                                       (df_spec_supp['mode']== mode) &
                                                       (df_spec_supp['algorithm'] == 'duc' )]['time'].values
            df_spec_supp_upd_time = df_spec_supp.loc[(df_spec_supp['support 2']==supp2) & 
                                                       (df_spec_supp['mode']== mode) &
                                                       (df_spec_supp['algorithm'] == 'updated' )]['time'].values

            rel_timechange = [y / x for x,y in zip(sorted(df_spec_supp_upd_time), 
                                                   sorted(df_spec_supp_duc_time))]
            if mode[0]=='F':
                abstraction = 'finance'
            else:
                abstraction = 'google'
            for rtime in rel_timechange:
                results.append([supp2, mode, rtime, abstraction])
                all_results.append([q1-q2, mode, rtime,abstraction])
    columns = ['New support', 'Abstraction', 'Rel timechange', 'Dataset' ]
    df_plot = pd.DataFrame(results, columns=columns)
    grid = sns.relplot(data=df_plot, x='New support', y='Rel timechange', hue='Abstraction',
                       col='Dataset', kind='line', style='Abstraction', markers=True)
    grid.set(yscale='log', yticks= [1,10], xticks=[0.75, 0.85, .95])
    grid.savefig('experiments/BTW2025/plots/spec_supp.pdf')

    columns = ['Change of Queryset size', 'Abstraction', 'Rel timechange', 'Dataset' ]
    df_plot = pd.DataFrame(all_results, columns=columns)
    grid = sns.relplot(data= df_plot, x= 'Change of Queryset size', y='Rel timechange', hue='Abstraction' )
    grid.set(yscale='log', yticks= [1,10,100])
    grid.savefig('experiments/BTW2025/plots/queryset_change1.pdf')
    dataframe['Change of Queryset size'] = dataframe['queryset size'] - dataframe['searchspace']
    grid = sns.relplot(data= dataframe, x= 'Change of Queryset size', y='time', hue='algorithm' )
    grid.savefig('experiments/BTW2025/plots/queryset_change.pdf')
if __name__ == "__main__":
    main()