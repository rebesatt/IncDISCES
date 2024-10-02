#!/usr/bin/python3
"""Contains functions for discovering queries from samples by using bottom up algorithms."""
import sys
import logging
import time
from collections import deque, Counter
from itertools import combinations, product, chain, combinations_with_replacement, permutations
from copy import deepcopy
from math import ceil
import numpy as np

sys.path.append('.')
from src.sample_multidim import MultidimSample
from src.query_multidim import MultidimQuery
from src.hyper_linked_tree import HyperLinkedTree



#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

def domain_unified_discovery_smarter(sample, supp, max_query_length) -> dict:
    """Query Discovery by using unified bottom up depth-first search with smarter matching.

    Args:
        sample: Sample instance.
        supp: Float between 0 and 1 which describes the requested support.

    Returns:
        Set of queries if a query has been discovered, None otherwise.
    """
    if max_query_length == -1:
        threshold = ceil(sample._sample_size * supp)
        trace_length = sorted([len(trace.split()) for trace in sample._sample])

        max_query_length = trace_length[sample._sample_size - threshold]
    matching_dict = {}
    non_matching_dict = {}
    domain_cnt = sample._sample_event_dimension
    alphabet = set()
    if supp == 1.0:
        _,min_trace_length= sample.get_sample_min_trace()
        max_query_length = min(max_query_length, min_trace_length)
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    att_vsdb = sample.get_att_vertical_sequence_database()
 
    sample_size = sample._sample_size
    vsdb = {}
    patternset ={}
    all_patternset = {}
    for domain, dom_vsdb in att_vsdb.items():
        patternset[domain] = set()
        all_patternset[domain] = {trace_id: set() for trace_id in range(sample_size)}
        for key, value in dom_vsdb.items():
            new_key = ''.join(gen_event_list[:domain] + [key] + gen_event_list[domain:])
            vsdb[new_key] = value

            for item in value.keys():
                if len(value[item]) >= 2:
                    all_patternset[domain][item].add(key)
                    patternset[domain].add(key)
                    # break


    sample_sized_support = ceil(sample._sample_size * supp)
    alphabet = {symbol for symbol,value in vsdb.items() if len(value) >= sample_sized_support}
    parent_dict = {}
    alphabet=sorted(alphabet)
    query = MultidimQuery()
    query.set_query_string(gen_event)
    querystring= query._query_string
    matching_dict[querystring] = query
    stack= deque()
    dict_iter = {}
    matching = True
    querycount=1
    dictionary= {}
    parent_dict[querystring] = query

    children = _next_queries_multidim(query,alphabet, max_query_length, patternset)
    parent_dict.update({child._query_string: query for child in children})
    stack.extend(children)
    query_tree = HyperLinkedTree(ceil(supp*sample._sample_size), event_dimension=sample._sample_event_dimension)

    start_time = time.time()
    last_print_time = start_time

    while stack:
        query = stack.pop()
        querystring = query._query_string
        querycount+=1
        current_time = time.time()

        if current_time - last_print_time > 300:
            LOGGER.info('Current query: %s; current stack size: %i; Current Query count: %i', querystring, len(stack), querycount)
            last_print_time = current_time
        parent = parent_dict[querystring]
        parentstring = parent._query_string
        matching= query.match_sample(sample = sample, supp= supp, dict_iter = dict_iter, patternset = all_patternset, parent_dict = parent_dict)
        dictionary.update({querystring:matching})

        if not matching:
            non_matching_dict[querystring] = query
            
        else:
            matching_dict[querystring] = query
            
            if parent_dict[querystring]._query_string == gen_event:
                parentstring = ''
            else:
                parentstring = parent_dict[querystring]._query_string
            parent_vertex = query_tree.find_vertex(parentstring)
            if not query_tree.find_vertex(querystring):
                vertex = query_tree.insert_query_string(parent_vertex, querystring, query=query, search_for_parents=False)
                vertex.matched_traces = query._query_matched_traces
            children = _next_queries_multidim(query,alphabet, max_query_length, patternset)
            if children:
                stack.extend(children)
                parent_dict.update({child._query_string: query for child in children})

    
    result_dict = {}
    result_dict['queryset'] = set(matching_dict.keys()) - {gen_event} - {''}
    result_dict['querycount'] =  querycount
    result_dict['parent_dict'] = parent_dict
    result_dict['matching_dict'] = matching_dict
    result_dict['dict_iter'] = dict_iter
    result_dict['query_tree'] = query_tree
    result_dict['non_matching_dict'] = non_matching_dict
    result_dict['patternset'] = all_patternset
    return result_dict


def _next_queries_multidim(query, alphabet, max_query_length, patternset, only_types = False):
    """Given a query the function calculates the children of that query which are the next queries that are more specialised adding one element following the rule set.

    Args:
        query (Query): an instance of Query
        alphabet (list): list of supported types
        max_query_length (int): maximal number of events in a query

    Returns:
        Children of the given query: list of query strings

    """
    querystring = query._query_string
    
    querylength= query._query_string_length
    querystring_list = query.get_query_list()
    domain_cnt = query._query_event_dimension
    variables=query._query_repeated_variables
    if variables:
        last_var = sorted(variables)[-1]
    # typeset = query._query_typeset
    num_of_vars= len(variables)
    gen_event= ';' * domain_cnt
    gen_event_list = [i for i in gen_event]
    if querystring == '':
        return [MultidimQuery(gen_event)]
    children=[]
    children_strings = set()
    pos_last_type = query._pos_last_type_and_variable[0]
    pos_first_var = query._pos_last_type_and_variable[1]
    pos_last_var = query._pos_last_type_and_variable[2]


    #special case: most general query
    if querystring == gen_event:
        if max_query_length>= 2 and not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var= ''.join(gen_event_list[:domain] + ['$x0'] + gen_event_list[domain:])
                    child=domain_var + ' ' + domain_var
                    child_query= MultidimQuery()
                    child_query._query_string = child
                    child_query._query_repeated_variables= {'x0'}
                    child_query._query_string_length= 2
                    child_query._pos_last_type_and_variable = np.array([-1, 0, 1])
                    child_query._query_event_dimension = query._query_event_dimension
                    children.append(child_query)

        if max_query_length>=1:
            for letter in alphabet:
                child=str(letter)
                child_query= MultidimQuery()
                child_query._query_string = child
                # child_query._query_typeset= typeset | {letter}
                child_query._query_string_length= 1
                child_query._pos_last_type_and_variable = np.array([0, -1,-1])
                child_query._query_event_dimension = query._query_event_dimension
                children.append(child_query)



    #non-empty querystrings
    else:
        
        #insert new variable twice: after last occurence of type and after first occurence of last variable
        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event= querystring_list[first_pos]
        first_pos_domains = query.non_empty_domain(first_pos_event)
        #first_pos_domain = domain_cnt
        for domain in first_pos_domains:
            att = first_pos_event.split(';')[domain]
            if first_pos == pos_last_type and att.count('$') == 0:
                first_pos_domain = domain
            if variables:
                if first_pos == pos_first_var and last_var in att:
                    first_pos_domain = domain

        querystring_split = querystring_list[first_pos:]
        if not only_types:
            for domain in range(domain_cnt):
                if patternset[domain]:
                    domain_var= ''.join(gen_event_list[:domain] + ['$x' + str(num_of_vars)] + gen_event_list[domain:])
                    var_domain = domain_var.find(domain_var.strip(';'))
                    var = 'x' + str(num_of_vars)
                    for idx, event in enumerate(querystring_split, start = first_pos):
                        if querylength+1 <= max_query_length:
                            if idx != querylength -1:
                                child = " ".join(querystring_list[:idx+1]) + ' '+domain_var +' ' + " ".join(querystring.split()[idx+1:])
                            else:
                                child = querystring + ' ' + domain_var

                            for idx2, event2 in enumerate(child.split()[idx+1:], start = idx+1):
                                if querylength+2 <= max_query_length:
                                    if idx2 != querylength:
                                        child2 = " ".join(child.split()[:idx2+1]) + ' '+domain_var +' ' + " ".join(child.split()[idx2+1:])
                                    else:
                                        child2 = child + ' ' + domain_var

                                    child_query= MultidimQuery()
                                    child_query._query_string = child2.strip()
                                    # child_query._query_typeset= typeset
                                    child_query._query_repeated_variables = variables |{var}
                                    child_query._query_string_length= querylength + 2
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx+1, idx2+1])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)

                                last_non_empty = query.non_empty_domain(event2)[-1]
                                if not event2.split(';')[var_domain] and idx2 <= querylength +1:
                                    #if  var_domain > last_non_empty:
                                    new_event = event2.split(';')
                                    new_event[var_domain] = domain_var.strip(';')
                                    # if idx == 0:
                                    #     child3 = ';'.join(new_event)
                                    if idx == querylength -1:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)
                                    else:
                                        child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(child.split()[idx2+1:])
                                    child_query= MultidimQuery()
                                    child_query._query_string = child3.strip()
                                    # child_query._query_typeset= typeset
                                    child_query._query_repeated_variables = variables |{var}
                                    child_query._query_string_length= querylength +1
                                    child_query._query_event_dimension = query._query_event_dimension
                                    child_query._pos_last_type_and_variable = np.array([pos_last_type, idx+1, idx2])
                                    assert child_query._query_string_length <= max_query_length
                                    if child_query._query_string not in children_strings:
                                        children.append(child_query)
                                        children_strings.add(child_query._query_string)


                        #last_non_empty = non_empty_domain(event)[-1]
                        if not event.split(';')[var_domain]:
                            if idx != first_pos or var_domain > first_pos_domain:
                                new_event = event.split(';')
                                new_event[var_domain] = domain_var.strip(';')
                                # if idx == 0:
                                #     child = ';'.join(new_event)
                                if idx == querylength -1:
                                    child = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                                else:
                                    child = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])

                                for idx2, event2 in enumerate(child.split()[idx:], start = idx):
                                    if querylength+1 <= max_query_length:
                                        if idx2 != querylength -1:
                                            child2 = " ".join(child.split()[:idx2+1]) + ' '+domain_var +' ' + " ".join(child.split()[idx2+1:])
                                        else:
                                            child2 = child + ' ' + domain_var
                                        child_query= MultidimQuery()
                                        child_query._query_string = child2.strip()
                                        # child_query._query_typeset= typeset
                                        child_query._query_repeated_variables = variables |{var}
                                        child_query._query_string_length= querylength + 1
                                        child_query._query_event_dimension = query._query_event_dimension
                                        child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                        assert child_query._query_string_length <= max_query_length

                                        if child_query._query_string not in children_strings:
                                            children.append(child_query)
                                            children_strings.add(child_query._query_string)

                                    last_non_empty = query.non_empty_domain(event2)[-1]
                                    if not event2.split(';')[var_domain]:
                                        if idx2 != first_pos: #or var_domain > last_non_empty:
                                            new_event = event2.split(';')
                                            new_event[var_domain] = domain_var.strip(';')
                                            # if idx == 0:
                                            #     child3 = ';'.join(new_event)
                                            if idx == querylength -1:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)
                                            else:
                                                child3 = " ".join(child.split()[:idx2]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(child.split()[idx2+1:])
                                            child_query= MultidimQuery()
                                            child_query._query_string = child3.strip()
                                            # child_query._query_typeset= typeset
                                            child_query._query_repeated_variables = variables |{var}
                                            child_query._query_string_length= querylength
                                            child_query._query_event_dimension = query._query_event_dimension
                                            child_query._pos_last_type_and_variable = np.array([pos_last_type, idx, idx2])
                                            assert child_query._query_string_length <= max_query_length

                                            if child_query._query_string not in children_strings:
                                                children.append(child_query)
                                                children_strings.add(child_query._query_string)


        #insert last inserted variable again
        if pos_first_var !=-1:
            var_numb=0
            for domain, letter in enumerate(querystring_list[pos_first_var].split(';')):
                if '$' in letter and var_numb <= int(letter.strip('$x;')):
                    last_variable_domain = domain
                    var_numb = int(letter.strip('$x;'))

            last_variable = querystring_list[pos_first_var].split(';')[last_variable_domain]
            num_of_vars= int(last_variable.strip('$x;')) +1
            domain_var= ''.join(gen_event_list[:last_variable_domain] + [last_variable] + gen_event_list[last_variable_domain:])
        else:
            last_variable = '$x0'
            num_of_vars= 0


        if pos_first_var>= pos_last_type:
            no_letter= True
            if pos_first_var == pos_last_type:
                for event in querystring_list[pos_first_var].split(';')[last_variable_domain+1:-1]:
                    if event.count('$') == 0 and event:
                        no_letter = False
            if no_letter and not only_types:
                first_pos = max(pos_last_type, pos_last_var)
                querystring_split = querystring_list[first_pos:]
                for idx, event in enumerate(querystring_split, start = first_pos):
                    if querylength+1 <= max_query_length:
                        if idx != querylength -1:
                            child = " ".join(querystring_list[:idx+1]) + ' '+domain_var +' ' + " ".join(querystring_list[idx+1:])
                        else:
                            child = querystring + ' ' + domain_var
                        child_query= MultidimQuery()
                        child_query._query_string = child.strip()
                        # child_query._query_typeset= typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength + 1
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx+1])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)
                    var_domain = domain_var.find(domain_var.strip(';'))
                    last_non_empty = query.non_empty_domain(event)[-1]
                    if not event.split(';')[var_domain]:
                        #if idx != first_pos or var_domain > last_non_empty:
                        new_event = event.split(';')
                        new_event[var_domain] = domain_var.strip(';')
                        if idx == 0:
                            child2 = ';'.join(new_event)
                        elif idx == querylength -1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])
                        child_query= MultidimQuery()
                        child_query._query_string = child2.strip()
                        # child_query._query_typeset= typeset
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, idx])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)


        #insert types: after last occurence of type and after first occurence of last variable
        first_pos = max(pos_last_type, pos_first_var)
        first_pos_event= querystring_list[first_pos]
        if 'last_variable_domain' in locals():
            last_symbol_domain = last_variable_domain
        else:
            for domain, letter in enumerate(first_pos_event.split(';')):
                if letter and '$' not in letter:
                    last_symbol_domain = domain

        querystring_split = querystring_list[first_pos:]
        for letter in alphabet:
            for idx, event in enumerate(querystring_split, start = first_pos):
                if querylength+1 <= max_query_length:
                    if idx != querylength -1:
                        child = " ".join(querystring_list[:idx+1]) + ' '+letter +' ' + " ".join(querystring_list[idx+1:])
                    else:
                        child = querystring + ' ' + letter
                    child_query= MultidimQuery()
                    child_query._query_string = child.strip()
                    # child_query._query_typeset= typeset | {letter}
                    child_query._query_repeated_variables = variables
                    child_query._query_string_length= querylength + 1
                    child_query._query_event_dimension = query._query_event_dimension
                    if idx < pos_last_var:
                        child_query._pos_last_type_and_variable = np.array([idx+1, pos_first_var, pos_last_var+1])
                    else:
                        child_query._pos_last_type_and_variable = np.array([idx+1, pos_first_var, pos_last_var])
                    assert child_query._query_string_length <= max_query_length
                    if child_query._query_string not in children_strings:
                        children.append(child_query)
                        children_strings.add(child_query._query_string)
                letter_domain = letter.find(letter.strip(';'))
                last_non_empty = query.non_empty_domain(event)[-1]
                if not event.split(';')[letter_domain]:
                    if idx != first_pos or letter_domain > last_non_empty or '$' in event.split(';')[last_non_empty]:
                        if letter_domain < last_symbol_domain and idx == first_pos:
                            continue
                        new_event = event.split(';')
                        new_event[letter_domain] = letter.strip(';')
                        # if idx == 0:
                        #     child2 = ';'.join(new_event)
                        if idx == querylength -1:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)
                        else:
                            child2 = " ".join(querystring_list[:idx]) + ' ' +  ';'.join(new_event)+ ' ' + " ".join(querystring_list[idx+1:])
                        child_query= MultidimQuery()
                        child_query._query_string = child2.strip()
                        # child_query._query_typeset= typeset | {letter}
                        child_query._query_repeated_variables = variables
                        child_query._query_string_length= querylength
                        child_query._query_event_dimension = query._query_event_dimension
                        child_query._pos_last_type_and_variable = np.array([idx, pos_first_var, pos_last_var])
                        assert child_query._query_string_length <= max_query_length

                        if child_query._query_string not in children_strings:
                            children.append(child_query)
                            children_strings.add(child_query._query_string)


    return children
