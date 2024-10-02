#!/usr/bin/python3
""" Contains the class for handling multidimensional Queries"""
import logging
import sys
from copy import copy
from typing import Pattern, Match
from typing_extensions import Self
from math import ceil
import numpy as np
sys.path.append('.')
from src.sample_multidim import MultidimSample


#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

#TODO (Docu): Sample instance -> MultidimSample instance

class MultidimQuery():
    """
        Base class for representing and discovering a multidimensional Query.

        Queries consist of a query string, global or local window size(s) and
        optional gap constraints.

        The query string consists of events which are composed of one or more
        attributes. The number of attributes defines the so called dimension of
        the corresponding event. Each attribute is represented by so called
        types or variables. Types are represented by strings over an alphabet
        (ascii, excl. whitespace, semicolon and $).

        Gap constraints describe which types are forbidden between to positions
        of the query string.

        The global window size denotes the range for a match in a trace, while
        the list of local window size tuples describes lower and upper bounds
        for the length of each gap between two consecutive events in the query
        string.
        Note that these bounds range over attributes instead of whole events.
        Given a query a;b; c;d; consisting of two 2-dimensional events we
        assume the local window sizes to be [(0,0),(i,j),(0,0)] and
            i,j mod 2 = 0, i<=j,
        since events are 2-dimensional. The a (0,0)-tuple at a gap within an
        event ensures that a; and b; (or c; and d;) correspond to the same
        event. A query in normalfrom may have up to two additional window size
        tuples, up to one left of the first and up to one right of the last
        event. We use the tuple (-1,-1) as a placeholder for these two.

        A query matches a trace t from a sample iff there exists a mapping from
        the query string to t, i.e. the query string is a subsequence of t, and
        neither the window size(s) nor the gap constraints are violated.

        Given a sample and a support this class offers the functionality to
        discover queries which fullfill the support.

        Attributes:
            _query_string: A query string consists of events represented by a
                number of types and/or variables, each symbolizing an attribute
                of the event. Events are modeled as blocks within
                _query_string, separated by whitespaces. The end of each
                attribute is marked by ";". The beginning of a variable is
                marked by $, every unmarked attribute is a type. Note that
                #; = #events x dimension and #events = #spaces - 1.

            _query_list: List of event strings which represents the query string.

            _query_event_dimension: Integer which represents the max. number of
                attributes per event.

            _query_string_length: Integer which represents the length of the
                query string, i.e. the number of events.

            _query_repeated_variables: Set of strings which represent the
                repeated variabels in the _query_string. A variable is called
                'repeated' if it occurs more than once in the _query_string.

            _query_matched_traces: List of indices of all traces in
                _query_sample, i.e. the positions of the traces within the
                sample, s.t. the a successful match test was performed. Note
                that this list may not contain all traces that match the query!

            _query_not_matched_traces: List of indices of all traces in
                _query_sample, i.e. the positions of the traces within the
                sample, s.t. an unsuccessful match test was performed. Note
                that this list may not contain all traces that do not match the
                query!

            _pos_last_type_and_variable: A numpy array containing last type
                position first position of last variable, last position of last
                variable. Default value: np.array([-1,-1,-1]).
    """
    def __init__(self, given_query_string=None, given_query_gap_constraints=None, given_query_windowsize_global=-1, given_query_windowsize_local=None, is_in_normalform=False) -> None:
        LOGGER.debug('Creating an instance of Query')
        self._query_repeated_variables:set = set()
        """Set of variables occuring more than once in _query_string. Default: set()"""
        self._query_attribute_typesets:dict = dict()
        """Dict storing the set of occuring types per attribute. Default: dict()"""
        self._query_list:list = []
        """List of event strings which represents the query string. Default: []"""

        if given_query_string is not None:
            assert isinstance(given_query_string, str)
            self._query_string:str = given_query_string
            """String of events consisting of types and variables. Events are separated by whitespaces. Default: ''"""
            self.set_query_string_length()
            self.set_query_repeated_variables()
            self.set_query_event_dimension()

        else:
            self._query_string = ""
            self._query_string_length:int = 0
            """Number of events within _query_string, i.e. number of types and variables. Default: 0"""
            self._query_event_dimension:int = 1
            """Max. number of attributes per event. Default: 1"""
            self._pos_last_type_and_variable = np.array([-1,-1,-1])
            """Np array containing last type pos, 1st pos of last var, last pos of last var. Default: np.array([-1,-1,-1])"""

        self._query_matched_traces:list = []
        """List of all trace-indices in _query_sample, s.t. a successful match test was performed. Default: []"""
        self._query_not_matched_traces:list = []
        """List of all trace-indices in _query_sample, s.t. an unsuccessful match test was performed. Default: []"""


    ##################################################


    def match_sample(self, sample, supp,  dict_iter = None,
                     patternset = None, parent_dict = None, max_query_length = -1):
        """
            Checks whether the query matches a given sample with given support.

            Determines and sets the support of the query regarding the sample
            if a full test is performed. Stores indices of tested traces
            depending on the match test result in _query_matched_traces or
            _query_not_matched_traces.

            Args:
                sample: Sample instance.

                supp: Float between 0 and 1 which describes the requested
                    support.
                
                dict_iter= dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None

                patternset: set of types occurring twice in at least one trace. Default None

                parent_dict: Dictionary containing parent query to each querystring. Default None

            Returns:
                True iff the query matches the given sample with given supp.

        """
        
        sample_size = sample._sample_size
        querystring = self._query_string
        if max_query_length != -1 and self._query_string_length > max_query_length:
            return False
        
        if self._query_matched_traces:
            trace_list = self._query_matched_traces
            for trace in trace_list:
                if trace < sample_size:
                    if not sample._sample[trace]:
                        self._query_matched_traces.remove(trace)
                        if querystring.count('$')!=0 and dict_iter:
                            dict_iter[querystring][trace]= -1
                        elif dict_iter:
                            dict_iter[querystring].pop(trace)
            if len(self._query_matched_traces)/sample_size >= supp:
                return True
            elif len(self._query_matched_traces) + sample_size - trace_list[-1] < ceil(supp*sample_size):
                return False

        matching = self._matching_smarter_multidim(sample=sample, supp =supp, dict_iter=dict_iter,
                                                        patternset=patternset,  parent_dict=parent_dict)
        if querystring.count('$')!=0:
            matchingcount= len(matching)
            self._query_matched_traces = list(matching.keys())
        else:
            matchingcount=0
            matched_traces = []
            for key, value in matching.items():
                if value != -1:
                    matched_traces.append(key)
                    matchingcount +=1
            self._query_matched_traces = matched_traces
        
        dict_iter[querystring] = matching
        matchsupport= matchingcount/sample_size
        if matchsupport< supp:
            return False
        else:            
            return True

    ##################################################

    
    def _matching_smarter_multidim(self, sample,supp, dict_iter, patternset,  parent_dict):
        """Matches a query against all traces in the sample.

        Args:
            sample: Sample instance.
            supp:Float between 0 and 1 which describes the requested support. Default: 1

            dict_iter (dictionary): nested dictionary for each query and trace 
                the last matching position is value. Default None

            patternset: set of types occurring twice in at least one trace. Default None

            parent_dict: Dictionary containing parent query to each querystring. Default None

        Returns:
            Trace dictionary containing trace index as keys and a dictionary of groups with the matched string and span as values.
        """
        querystring = self._query_string
        query_list = self.get_query_list()
        trace_split_list = sample.get_sample_list_split()
        if querystring in parent_dict:
            parent = parent_dict[querystring]
        else:
            parent = self._parent()
            parent.set_query_matchtest('smarter')
            parent_dict[querystring] = parent
        parentstring = parent._query_string
        parent_list = parent.get_query_list()
        trace_matches={}
        sample_size = sample._sample_size
        
        if not self._query_matched_traces:
            traces_to_match = list(range(sample_size))
            matched_traces = []
        else:
            matched_traces = self._query_matched_traces
            traces_to_match = list(range(matched_traces[-1]+1, sample_size))
            for trace in self._query_matched_traces:
                if trace in dict_iter[querystring]:
                    trace_matches[trace]= dict_iter[querystring][trace]

        if querystring.count('$x') == 0:
            num_trace_match = len(traces_to_match) + len(matched_traces)
            for trace_idx in traces_to_match:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace_idx]:
                    idx = -1
                else:
                    idx = self._smart_trace_match_multidim(querystring ,trace_split_list[trace_idx], trace_idx, dict_iter, query_list)
                if trace_idx not in trace_matches:
                    trace_matches[trace_idx]= {}
                trace_matches[trace_idx]= idx
                if idx == -1:
                    num_trace_match -=1

            return trace_matches

        var_count = querystring.count('$x')
        cur_count = 0
        var_int = -1
        for event in query_list:
            for dom, letter in enumerate(event.split(';')[:-1]):
                if '$x' in letter:
                    if int(letter[2:]) > var_int:
                        var_int = int(letter[2:])
                        var_domain = dom
                    cur_count +=1
                    if cur_count == var_count:
                        break

        if parentstring.count('$')==0:
            if not parentstring:
                parent_traces = list(range(sample_size))
            else:
                if parentstring in dict_iter:
                    parent_traces= list(dict_iter[parentstring].keys())
                    for stream in traces_to_match:
                        if stream not in parent_traces:
                            parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                            dict_iter[parentstring] = parent_match
                            parent_traces= list(parent_match.keys())
                            break
                else:
                    parent_match = parent._matching_smarter_multidim(sample,supp,  dict_iter, patternset, parent_dict)
                    dict_iter[parentstring] = parent_match
                    parent_traces= list(parent_match.keys())

            trace_list= [trace for trace in traces_to_match if trace in parent_traces]
            num_trace_match = len(trace_list) + len(matched_traces)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace]:
                    num_trace_match-=1
                    continue

                for letter in patternset[var_domain][trace]:
                    letter_querystring = querystring.replace('$x0', letter)
                    if letter_querystring in dict_iter:
                        if trace in dict_iter[letter_querystring]:
                            if sample._sample[trace]:
                                if dict_iter[letter_querystring][trace] !=-1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][(letter,)]= dict_iter[letter_querystring][trace]
                        else:
                            idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][(letter,)]= idx
                    else:
                        idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                        if idx != -1:
                            if trace not in trace_matches:
                                trace_matches[trace]= {}
                            trace_matches[trace][(letter,)]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

        else:
            parent_set = set()
            for event in parent_list:
                for symbol in event.split(';'):
                    if symbol.count('$') !=0:
                        parent_set.add(symbol[1:])
            parent_variables=sorted(list(parent_set))
            if parentstring in dict_iter:
                parent_traces= list(dict_iter[parentstring].keys())
                for stream in traces_to_match:
                    if stream not in parent_traces:
                        parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                        dict_iter[parentstring] = parent_match
                        parent_traces= list(parent_match.keys())
                        break
                
            else:
                parent_match = parent._matching_smarter_multidim(sample, supp, dict_iter, patternset, parent_dict)
                dict_iter[parentstring] = parent_match
                parent_traces= list(parent_match.keys())
            
            trace_list= [trace for trace in traces_to_match if trace in parent_traces]

            trace_list = set(traces_to_match) & set(parent_traces)
            num_trace_match = len(trace_list) + len(matched_traces)
            for trace in trace_list:
                if num_trace_match/sample_size < supp:
                    break
                if not sample._sample[trace]:
                    num_trace_match-=1
                    continue
                group_list= list(dict_iter[parentstring][trace].keys())
                for group in group_list:
                    letter_querystring=querystring
                    assert len(group) == len(parent_variables)
                    for val, letter in enumerate(group):
                        letter_querystring = letter_querystring.replace(f'${parent_variables[val]}', letter)
                    if letter_querystring.count('$')>0:

                        for letter in patternset[var_domain][trace]:
                            letter_querystring2 = letter_querystring.replace(f'$x{len(group)}', letter)
                            if letter_querystring2 in dict_iter:
                                if trace in dict_iter[letter_querystring2]:
                                    if dict_iter[letter_querystring2][trace] !=-1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= dict_iter[letter_querystring2][trace]

                                else:
                                    idx = self._smart_trace_match_multidim(letter_querystring2, trace_split_list[trace], trace, dict_iter)
                                    if idx != -1:
                                        if trace not in trace_matches:
                                            trace_matches[trace]= {}
                                        trace_matches[trace][group + (letter,)]= idx

                            else:
                                idx = self._smart_trace_match_multidim(letter_querystring2, trace_split_list[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group + (letter,)]= idx

                    else:
                        if letter_querystring in dict_iter:
                            if trace in dict_iter[letter_querystring]:
                                if dict_iter[letter_querystring][trace] !=-1 and sample._sample[trace]:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= dict_iter[letter_querystring][trace]
                            else:
                                idx= self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                                if idx != -1:
                                    if trace not in trace_matches:
                                        trace_matches[trace]= {}
                                    trace_matches[trace][group]= idx
                        else:
                            idx = self._smart_trace_match_multidim(letter_querystring, trace_split_list[trace], trace, dict_iter)
                            if idx != -1:
                                if trace not in trace_matches:
                                    trace_matches[trace]= {}
                                trace_matches[trace][group]= idx

                if trace not in trace_matches:
                    num_trace_match-=1
            return trace_matches

    def _smart_trace_match_multidim(self, querystring, trace_split, trace_idx, dict_iter, query_split=None):
        """Given a trace and a querystring the matching position is calculated and in case of a match
        the dict_iter is updated.

        Args:
            querystring (String): querystring for query
            trace (List): trace list from the sample
            trace_idx (int): index of the given trace
            dict_iter (dictionary): nested dictionary for each query and trace the last matching position is value.

        Returns:
            Last matching position as integer. -1 if there
            is no match.
        """
        if not query_split:
            query_split = querystring.split()
        domain_cnt= query_split[0].count(';')
        gen_event= ';' * domain_cnt
        last_event= query_split[-1]
        non_empty_domains = self.non_empty_domain(last_event)
        parentstring = ' '.join(query_split[:-1])
        last_event_split = last_event.split(';')
        if non_empty_domains:
            last_non_empty = non_empty_domains[-1]
        if not parentstring:
            if len(non_empty_domains) <=1:
                parentstring = gen_event
            else:
                parentstring= ';'.join(last_event_split[:last_non_empty])+ ';' + ';'.join(last_event_split[last_non_empty+1:]) +';'

        else:
            if len(non_empty_domains)> 1:
                parentstring = parentstring + ' ' + ';'.join(last_event_split[:last_non_empty])+ ';' + ';'.join(last_event_split[last_non_empty+1:]) +';'
        if querystring not in dict_iter:
            dict_iter[querystring]= {}
        if querystring == gen_event:
            dict_iter[querystring][trace_idx]=0
            return 0
        if parentstring not in dict_iter:
            idx = self._smart_trace_match_multidim(parentstring,trace_split, trace_idx, dict_iter)
        if trace_idx in dict_iter[parentstring]:
            parent_end_pos = dict_iter[parentstring][trace_idx]
        else:
            idx= self._smart_trace_match_multidim(parentstring, trace_split, trace_idx, dict_iter)
            parent_end_pos = dict_iter[parentstring][trace_idx]
        if parentstring == gen_event:
            domain_trace_split = [event.split(';')[last_non_empty] for event in trace_split]
            domain_type = last_event_split[last_non_empty]
            if domain_type in domain_trace_split:
                end_pos = domain_trace_split.index(domain_type)
            else:
                end_pos = -1
            dict_iter[querystring][trace_idx]= end_pos
            return end_pos
        if parent_end_pos !=-1:
            if len(non_empty_domains) == 1:
                domain_trace_split = [event.split(';')[last_non_empty] for event in trace_split]
                domain_type= last_event_split[last_non_empty]


                domain_trace_list = domain_trace_split[parent_end_pos+1:]

                if domain_type and domain_type in domain_trace_list:
                    idx = domain_trace_list.index(domain_type)
                else:
                    idx = -1
            else:
                if trace_split[parent_end_pos].split(';')[last_non_empty] and trace_split[parent_end_pos].split(';')[last_non_empty] == last_event_split[last_non_empty]:
                    end_pos = parent_end_pos
                    dict_iter[querystring][trace_idx]= end_pos
                    return end_pos
                else:
                    remaining_trace = ' '.join(trace_split[parent_end_pos+1:])
                    remaining_trace_split = remaining_trace.split()
                    for i, dom in enumerate(non_empty_domains):
                        domain_trace_split = [event.split(';')[last_non_empty] for event in remaining_trace_split]

                        domain_type= last_event_split[dom]
                        domain_trace_list = domain_trace_split
                        if domain_type in domain_trace_list:
                            idx_list = {i for i, ltr in enumerate(domain_trace_list) if ltr == domain_type}
                        else:
                            idx = -1
                            break
                        if i == 0:
                            index_set = idx_list
                        else:
                            index_set = index_set & idx_list
                            if not index_set:
                                idx = -1
                                break
                        if dom == last_non_empty:
                            idx = min(index_set)

            if idx != -1:
                if trace_idx not in dict_iter[querystring]:
                    dict_iter[querystring][trace_idx]= {}
                end_pos = dict_iter[parentstring][trace_idx] + idx +1
                dict_iter[querystring][trace_idx]= end_pos
                return end_pos
            else:
                dict_iter[querystring][trace_idx]=-1
                return -1
        else:
            dict_iter[querystring][trace_idx]=-1
            return -1
    
    ##################################################

    def _parent(self):
        """Returns the parent query of a given Multidimquery,
            in other words the Multidimquery from which it was generated
            acording to the constrain-based rule set.
            query (MultidimQuery): an instance of MultidimQuery

        Returns:
            parent (MultidimQuery): an instance of MultidimQuery
        """
        querystring= self._query_string
        query_list = self.get_query_list()
        if not querystring:
            return MultidimQuery()

        var_int = -1
        pos_first_var = -1
        pos_last_var = -1
        pos_last_type = -1
        for pos, event in enumerate(query_list):
            for dom, letter in enumerate(event.split(';')[:-1]):
                if '$x' in letter:
                    if int(letter[2:]) > var_int:
                        var_int = int(letter[2:])
                        pos_first_var = pos
                    elif int(letter[2:]) == var_int:
                        pos_last_var = pos
                elif letter:
                    pos_last_type = pos
                    

        var = False
        letter = False
        domain_cnt = query_list[0].count(';')
        gen_event = ';' *domain_cnt
        if pos_last_type > pos_first_var:
            last_position = pos_last_type
            letter = True

        elif pos_last_type < pos_first_var:
            last_position = pos_last_var
            var = True
        else:
            last_position = pos_last_type
            current_event = query_list[last_position]
            filled_domains = self.non_empty_domain(current_event)
            last_domain =filled_domains[-1]
            last_type = current_event.split(';')[last_domain]
            if last_type.count("$") > 0:
                last_position = pos_last_var
                var = True
            else:
                letter = True

        current_event = query_list[last_position]
        filled_domains = self.non_empty_domain(current_event)
        if var:
            var_numb = -1
        for domain in filled_domains:
            att = current_event.split(';')[domain]
            if letter and '$' not in att:
                last_domain= domain
            if var and '$' in att and int(att.strip('$x')) > var_numb:
                last_domain = domain
                var_numb = int(att.strip('$x'))
        #last_domain =filled_domains[-1]
        last_type = current_event.split(';')[last_domain]
        if len(filled_domains) >1:
            current_event_split = current_event.split(';')
            current_event_split[last_domain] = ''
            current_event = ';'.join(current_event_split)
            #current_event= ';'.join(current_event.split(';')[:last_domain]) +';' + ';'.join(current_event.split(';')[last_domain+1:]) +';'
        else:
            current_event = gen_event

        if current_event != gen_event:
            parentstring = " ".join(query_list[:last_position]) + " "+ current_event + " " + " ".join(query_list[last_position+1:])
        else:
            parentstring = " ".join(query_list[:last_position]) + " "+ " ".join(query_list[last_position+1:])
        if last_type.count('$') !=0:
            if parentstring.count(last_type) == 1:
                last_position = pos_first_var
                current_event = query_list[last_position]
                current_event= current_event.replace(last_type, '')

                if current_event != gen_event:
                    parentstring = " ".join(parentstring.split()[:last_position]) + " "+ current_event + " " + " ".join(parentstring.split()[last_position+1:])
                else:
                    parentstring = " ".join(parentstring.split()[:last_position]) + " "+ " ".join(parentstring.split()[last_position+1:])

        parent=MultidimQuery()
        parentstring = parentstring.strip()
        if parentstring and parentstring != gen_event:
            parent.set_query_string(parentstring.strip(), recalculate_attributes=False)

        return parent


    def non_empty_domain(self, last_event):
        """Returns the number of the last non empty domain.

        Args:
            querystring (String)

        Returns:
            List of domain-numbers that are not empty.
        """
        last_event_split = last_event.split(';')
        non_empty_domains= [idx for idx, i in enumerate(last_event_split) if i]

        return non_empty_domains



    ##################################################


    def set_query_string(self, querystring:str, recalculate_attributes:bool=True, to_regex:bool=True) -> None:
        """
            Sets _query_string to querystring and does some updates.

            Updates _query_string_length and _query_repeated_variables as well
            to avoid an inconsistent query.

            Args:
                querystring: A string which represents the new query string. We
                    assume that querystring fits to the described format of a
                    query string.

                recalculate_attributes: Bool which indicates whether the other
                    query attributes should be updated. Sets them to default
                    values if recalculate_attributes is False.

                to_regex: Bool which indicates whether _query_string_regex
                    should be updated. Used True as default but is set to False
                    during a function call of query_string_to_normalform to
                    save runtime.
        """
        assert isinstance(querystring, str)
        self._query_string = querystring

        if recalculate_attributes is True:
            self.set_query_string_length()
            self.set_query_event_dimension()
            # self.set_query_typeset() #sets query_event_dimension as well
            self.set_query_repeated_variables()
        else:
            self._query_string_length = 0
            self._query_event_dimension = 1
            self._query_repeated_variables = set()

    def set_query_string_length(self) -> None:
        """
            Determines and sets the length of the query string.

            Note that the query string length equals the number of events which
            occur within the string.
        """
        if self._query_string == "":
            self._query_string_length = 0
        else:
            self._query_string_length = self._query_string.count(' ')+1


    def set_query_event_dimension(self) -> None:
        """
            Determines and set the maximum event dimension.

            Usually we assume that all events have the same dimension.
        """
        query_string_copy = copy(self._query_string)
        if query_string_copy == "":
            return
        dimension = -1
        while len(query_string_copy)>0:
            if query_string_copy[0] == " ":
                query_string_copy = query_string_copy[1:]
            else:
                event = query_string_copy.split(' ')[0]
                event_dimension = event.count(";")
                if event_dimension>dimension:
                    dimension = event_dimension
                query_string_copy = query_string_copy[len(event):]

        self._query_event_dimension = dimension

    def set_query_repeated_variables(self) -> None:
        """
            Determines and sets the set of repeated variables.

            Only variable names are stored, hence entries of the set will not
            start with '$'. Since _query_repeated_variables is a set, the
            variables are not ordered. The set will be empty if the length of
            the query string is 0, i.e. no query string is set.
        """
        self._query_repeated_variables.clear()
        variable_dict = {}

        if len(self._query_string) == 0:
            self._query_repeated_variables = set()

        query_string_copy = copy(self._query_string)
        while len(query_string_copy) > 0:
            if query_string_copy[0] == "$":
                query_string_copy = query_string_copy[1:]
                variable = query_string_copy.split(';')[0]
                if variable in variable_dict:
                    variable_dict[variable] = variable_dict[variable]+1
                else:
                    variable_dict[variable] = 1
                query_string_copy = query_string_copy[len(variable):]
            query_string_copy = query_string_copy[1:]

        for elem in variable_dict:
            if variable_dict[elem] > 1:
                self._query_repeated_variables.add(elem)


    def set_pos_last_type_and_variable(self) -> None:
        """
            Returns last position of a type and first and last position of the
            last variable.

            Args:
                query: an instance of Query

            Returns: three args
                last type position (int), first position of last variable (int)
                and last position of last variable (int).
        """
        querystring = self._query_string
        if not querystring:
            self._pos_last_type_and_variable = np.array([-1,-1,-1])
            return

        variables_set = set()
        querystring_split = querystring.split()
        for event in querystring_split:
            for symbol in event.split(';'):
                if symbol.count('$') !=0:
                    variables_set.add(symbol[1:])
        variables=sorted(list(variables_set))

        querylength = len(querystring_split)
        if querystring.count(';') == 0:
            if len(self._query_repeated_variables) !=0:
                pos_last_var= querylength - 1 - querystring_split[::-1].index('$'+variables[-1])
                pos_first_var= querystring_split.index('$x'+str(variables[-1][1]))

                if len(querystring_split) == querystring.count('$x'):
                    pos_last_type=-1
                    self._pos_last_type_and_variable = np.array([-1, pos_first_var, pos_last_var])
                else:
                    position=-1
                    while querystring_split[position].count('$x')!=0:
                        position-=1

                    pos_last_type= querylength + position
                    self._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, pos_last_var])

            else:
                pos_first_var=-1
                pos_last_var =-1

                self._pos_last_type_and_variable = np.array([querylength-1, pos_first_var, pos_last_var])
        else:
            xcount = querystring.count('$x')
            if xcount !=0:
                last_var= variables[-1]
                string_pos = querystring.find(last_var)
                pos_first_var = querystring[:string_pos].count(' ')
                string_pos2= querystring.rfind(last_var)
                pos_last_var= querystring[:string_pos2].count(' ')

                if len(querystring_split) == xcount:
                    pos_last_type=-1
                    self._pos_last_type_and_variable = np.array([-1, pos_first_var, pos_last_var])
                    return
                else:
                    position=-1
                    no_letter = True
                    while no_letter and position >= -querylength:
                        for event in querystring_split[position].split(';')[:-1]:
                            if event.count('$') == 0 and event:
                                no_letter = False
                                break
                        if no_letter:
                            position-=1

                    pos_last_type= querylength + position
                    self._pos_last_type_and_variable = np.array([pos_last_type, pos_first_var, pos_last_var])
                    return

            else:
                pos_first_var=-1
                pos_last_var =-1

                self._pos_last_type_and_variable = np.array([querylength-1, pos_first_var, pos_last_var])
                return

    ##################################################
    
    def get_query_list(self) -> list:
        """
            Returns a list of the query string.

            Returns:
                List of strings.
        """
        if not self._query_string:
            return []
        if not self._query_list:
            self._query_list = self._query_string.split()
        return self._query_list

##############################################################################
