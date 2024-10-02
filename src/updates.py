"""
Provides algorithms to update query sets after adding/deleting a stream or changing the global support.
"""
from math import ceil
import sys
from collections import deque
import time
import gzip
from copy import deepcopy

sys.path.append('.')
from src.discovery_bu_multidim import _next_queries_multidim, domain_unified_discovery_smarter
from src.sample_multidim import MultidimSample



def main():
    sample_path = 'datasets/google_query1_status1.txt.gz'
    trace_length = 7
    sample_size = 1000
    max_query_length = 4
    counter = 0
    sample_list = []
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
    sample2 = MultidimSample(sample_list)
    sample1 = MultidimSample(sample_list)
    copy_sample1 = deepcopy(sample1)
    copy_sample2 = deepcopy(sample2)
    supp1 = 0.8
    supp2 = 0.9
    
    result_dictionary = None
    result_dictionary = domain_unified_discovery_smarter(sample1, supp1, max_query_length=max_query_length)
    
    starttime = time.time()
    if result_dictionary is not None:
        
        copy_result_dictionary = deepcopy(result_dictionary)
        new_result_dictionary = update_query_set(copy_result_dictionary, copy_sample1, copy_sample2, supp1, supp2, max_query_length=max_query_length)

        print('Updated Algorithm Time:', time.time()-starttime)
    
    starttime = time.time()
    copy_sample2 = deepcopy(sample2)
    result_dictionary2 = domain_unified_discovery_smarter(copy_sample2, supp2, max_query_length=max_query_length)

    print('Baseline Algorithm Time:', time.time()-starttime)
    
    assert set(new_result_dictionary['matching_dict'].keys()) == set(result_dictionary2['matching_dict'].keys())

def update_query_set(result_dictionary:dict, sample1:MultidimSample, sample2:MultidimSample|None, supp1:float, supp2:float|None, max_query_length:float = -1) -> dict:
    """
    Update the query set after adding/deleting a stream or changing the global support.
    """
    # Update the query set
    if sample2 is None:
        sample2 = sample1
        sample1_size = sample1._sample_size
        sample2_size = sample2._sample_size
    else:
        # Check if the they are sample1 and sample2 are neighboring samples
        sample1_size = sample1._sample_size
        sample2_size = sample2._sample_size

    if supp2 is None:
        supp2 = supp1

    
    else:
        # Check if the new support is valid
        if supp1 <= 0 or supp1 > 1 or supp2 <= 0 or supp2 > 1:
            raise ValueError("The support must be between 0 and 1.")
        if supp2 < supp1 and sample2_size  > sample1_size:
            raise ValueError("When adding streams, the new support must be greater than or equal to the old support.")
        if supp2 > supp1 and sample2_size < sample1_size:
            raise ValueError("When deleting streams, the new support must be less than or equal to the old support.")
        
    if supp1 == supp2 and sample1_size == sample2_size:
        return result_dictionary
    
    # Check if it is a valid setting where s2 >= s1
    s1 = ceil(sample1_size * supp1)
    s2 = ceil(sample2_size * supp2)

    
    if sample1_size > sample2_size:
        if s1 < s2 + sample1_size-sample2_size:
            raise ValueError("This combination of sample sizes and support is not valid for the current algorithm. Please try a different combination.")
    
        new_result_dictionary = _update_query_set_specification(result_dictionary, sample1, sample2, supp1, supp2, max_query_length)
    elif sample1_size < sample2_size:
        if s2 < s1 + sample2_size-sample1_size:
            raise ValueError("This combination of sample sizes and support is not valid for the current algorithm. Please try a different combination.")
    
        new_result_dictionary = _update_query_set_generalisation(result_dictionary, sample1, sample2, supp1, supp2)
    else:
        if supp1 > supp2:
            new_result_dictionary = _update_query_set_specification(result_dictionary, sample1, sample2, supp1, supp2,max_query_length)
        else:
            new_result_dictionary = _update_query_set_generalisation(result_dictionary, sample1, sample2, supp1, supp2)
    
    return new_result_dictionary



def _update_query_set_specification(result_dictionary:dict, sample1:MultidimSample, sample2:MultidimSample, supp1:float, supp2:float, max_query_length:float) -> dict:
    """
    Update the query set after deleting a stream and/or decreasing the support value.
    """
    # Update the query set
    
    sample1_set = set(sample1._sample)
    sample2_set = set(sample2._sample)
    sample1_size = sample1._sample_size
    sample2_size = sample2._sample_size

    s1 = ceil(sample1._sample_size * supp1)
    s2 = ceil(sample2._sample_size * supp2)
    att_vsdb1 = sample1.get_att_vertical_sequence_database()
    att_vsdb2 = sample2.get_att_vertical_sequence_database()
    patternset ={}
    vsdb1 = {}
    vsdb2 = {}
    domain_cnt = sample1._sample_event_dimension
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    patternset = result_dictionary['patternset']
    
    for domain, dom_vsdb in att_vsdb1.items():
        for key, value in dom_vsdb.items():
            new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
            vsdb1[new_key] = value

    for domain, dom_vsdb in att_vsdb2.items():
        for key, value in dom_vsdb.items():
            new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
            vsdb2[new_key] = value
    
    alphabet1 = {symbol for symbol,value in vsdb1.items() if len(value) >= s1}
    alphabet2 = {symbol for symbol,value in vsdb2.items() if len(value) >= s2}

    stream_diff_id_list = []
    if sample2_size != sample1_size:
        stream_difference = sample1_set - sample2_set
        sample_id = -1
        counter = 0
        stream_diff_len = len(stream_difference)
        for i in range(sample1_size):
            if sample1._sample[i] in stream_difference:
                sample_id = i
                stream_diff_id_list.append(i)
                counter +=1
                if counter == stream_diff_len:
                    break
        sample2._sample = sample2._sample[:sample_id] + [''] + sample2._sample[sample_id+1:]
        sample2._sample_size = len(sample2._sample)
    else:
        stream_difference = {}

    
    trace_length = sorted([len(trace.split()) for trace in sample2._sample])
    max_query_length = min( trace_length[sample2._sample_size - s2], max_query_length)
    if max_query_length <0:
        max_query_length = trace_length[sample2._sample_size - s2]
    elif max_query_length > trace_length[sample2._sample_size - s2]:
        max_query_length = trace_length[sample2._sample_size - s2]
    dict_iter = result_dictionary['dict_iter']
    matching_dict = result_dictionary['matching_dict']
    parent_dict = result_dictionary['parent_dict']
    query_tree = result_dictionary['query_tree']
    non_maching_dict = {}
    
    stack = deque()
    stack.extend([query for query in result_dictionary['non_matching_dict'].values()])
    while stack:
        query = stack.pop()
        querystring = query._query_string
        if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
        else:
            parentstring = parent_dict[querystring]._query_string
        matching = query.match_sample(sample=sample2, supp=supp2, 
                                          dict_iter=dict_iter, patternset= patternset, 
                                          parent_dict=parent_dict, max_query_length=max_query_length)
        if matching:
            matching_dict[query._query_string] = query
            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
             query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
            
            
            children = _next_queries_multidim(query, alphabet1, patternset=patternset, max_query_length=max_query_length)
            if children:
                stack.extend(children)
                parent_dict.update({child._query_string: query for child in children})
        else:
            non_maching_dict[query._query_string] = query
    
    alphabet_diff = alphabet2 - alphabet1
    if alphabet_diff:
        stack2 = deque()
        seen = set()
        for new_type in alphabet_diff:
            seen_parent = set()
            querystring = gen_event
            query = matching_dict[querystring]
            new_type_set = set()
            new_type_set.add(new_type)
            children= _next_queries_multidim(query,new_type_set, patternset=patternset, 
                                             max_query_length=max_query_length, only_types=True)
            for child_query in children:
                parent_dict[child_query._query_string] = matching_dict[gen_event]
            if children:
                stack.extend(children)
            while stack:
                query = stack.pop()
                querystring = query._query_string
                if querystring in seen:
                    continue
                else:
                    seen.add(querystring)
                if parent_dict[querystring]._query_string == gen_event:
                    parentstring = ''
                else:
                    parentstring = parent_dict[querystring]._query_string
                matching = query.match_sample(sample=sample2, supp=supp2, 
                                            dict_iter=dict_iter, patternset= patternset, 
                                            parent_dict=parent_dict, max_query_length=max_query_length)
                if matching:
                    stack2.append(query)
                    matching_dict[query._query_string] = query
                    if parentstring in seen_parent:
                        continue
                    else:
                        seen_parent.add(parentstring)
                    parent_vertex = query_tree.find_vertex(parentstring)
                   
                    children_vertices = parent_vertex.child_vertices
                    for child_vertex in children_vertices:
                        child_query = child_vertex.query
                        grand_children = _next_queries_multidim(child_query,new_type_set, patternset=patternset, 
                                                                max_query_length=max_query_length, only_types=True)
                        if grand_children:
                            stack.extend(grand_children)
                            parent_dict.update({child._query_string: child_query for child in grand_children})
                else:
                    non_maching_dict[query._query_string] = query


        while stack2:
            query = stack2.pop()
            querystring = query._query_string
            if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
            else:
                parentstring = parent_dict[querystring]._query_string
            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
                query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
            matching = query.match_sample(sample=sample2, supp=supp2, 
                                          dict_iter=dict_iter, patternset= patternset, parent_dict=parent_dict)
            if matching:
                matching_dict[query._query_string] = query
                parent_vertex = query_tree.find_vertex(parentstring)
                if not query_tree.find_vertex(querystring):
                    query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
                
                children = _next_queries_multidim(query,alphabet2, patternset=patternset, max_query_length=max_query_length)
                if children:
                    stack2.extend(children)
                    parent_dict.update({child._query_string: query for child in children})
            else:
                non_maching_dict[query._query_string] = query       
        

    result_dictionary["queryset"] = set(matching_dict.keys()) - {gen_event} - {''}
    result_dictionary["non_matching_dict"] = non_maching_dict
    result_dictionary["matching_dict"] = matching_dict
    result_dictionary["dict_iter"] = dict_iter
    result_dictionary["parent_dict"] = parent_dict
    result_dictionary["query_tree"] = query_tree

    return result_dictionary

def _update_query_set_generalisation(result_dictionary:dict, sample1:MultidimSample, sample2:MultidimSample, supp1:float, supp2:float) -> dict:
    """
    Update the query set after adding a stream and/or increasing the support value.
    """
    # Update the query set

    sample1_set = set(sample1._sample)
    sample2_set = set(sample2._sample)
    sample1_size = sample1._sample_size
    sample2_size = sample2._sample_size
    stream_difference = sample2_set - sample1_set
    s2 = ceil(sample2_size * supp2)
    domain_cnt = sample1._sample_event_dimension
    gen_event= ';' * domain_cnt

    patternset = result_dictionary['patternset']
    att_vsdb2 = sample2.get_att_vertical_sequence_database()
    new_trace_ids = [sample2._sample.index(trace) for trace in stream_difference]

    for domain, dom_vsdb in att_vsdb2.items():
        patternset[domain].update({trace_id: set() for trace_id in new_trace_ids})
        for key, value in dom_vsdb.items():

            for item in new_trace_ids:
                if item in value:
                    if len(value[item]) >= 2:

                        patternset[domain][item].add(key)
        

    dict_iter = result_dictionary['dict_iter']
    matching_dict = result_dictionary['matching_dict']
    parent_dict = result_dictionary['parent_dict']
    query_tree = result_dictionary['query_tree']
    non_matching_set = set()

    queryset = result_dictionary['queryset']
    query_dict = {}
    
    stack = deque()
    query_dict[gen_event] = matching_dict[gen_event]
    vertex = query_tree.find_vertex('')
    children_querystrings = [child_vertex.query_string for child_vertex in vertex.child_vertices]
    if children_querystrings:
        stack.extend(children_querystrings)
    while stack:
        querystring = stack.pop()
        if querystring not in matching_dict:
            continue

        query = matching_dict[querystring]
        vertex = query_tree.find_vertex(querystring)
        matching = query.match_sample(sample=sample2, supp=supp2, 
                                    dict_iter=dict_iter, patternset= patternset, parent_dict=parent_dict)  
        if matching:
            query_dict[querystring] = query
            children_querystrings = [child_vertex.query_string for child_vertex in vertex.child_vertices]
            if children_querystrings:
                stack.extend(children_querystrings)

        else:
            non_matching_set.add(querystring)

    result_dictionary['matching_dict'] = query_dict
    result_dictionary['parent_dict'] = parent_dict
    return result_dictionary

if __name__ == "__main__":
    main()