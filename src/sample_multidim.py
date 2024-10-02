#!/usr/bin/python3
""" Contains the class for handling Samples"""
import logging
from copy import deepcopy
from math import ceil

#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class MultidimSample():
    """
        Base class for representing a Sample.

        Attributes:
            _sample: List of strings. Each string is called a trace and
            consists of events, separated by a whitespace. Events may be
            multidimensional and attributes (i.e. the single types of the
            event) are separated by semicolons.

            _sample_list_split: List of lists. Each list represents a trace
            and contains the events of the trace as lists of strings.

            _sample_size: Number of traces within the _sample.

            _sample_event_dimension: Integer which represents the max. number
            of attributes per event.

            _sample_typeset: Set of types occurring within the _sample. This
            can be seen as the alphabet for query discovery, since a query can
            not match a sample while containing further types.

            _sample_att_typesets: Mapping of attributes to typesets, stored as
            a dict, i.e. the set at index 0 contains all types occuring as a
            first type / attribute value of some event.

            _sample_supported_typeset: Dict mapping support thresholds to a
            corresponding set of types occurring within the _sample satisfying
            the given support threshold.

            _sample_att_supported_typeset: Dict mapping attributes to typesets,
            s.t. each type in a set satisfies the given support for the corres-
            ponding attribute.

            _sample_vertical_sequence_database: contains a tracewise
            vertical map from type to occurence in trace

            _dim_sample_dict: Dictionary with dimensions as key and corres-
            ponding one-dim Sample as value.
    """
    def __init__(self, given_sample=None, uniform_dimension=False) -> None:
        LOGGER.debug('Creating an instance of Sample')
        ### private
        if not given_sample:             # checks if given_sample == None:
            given_sample = []
        self._sample:list = given_sample
        """List of strings representing traces. Default: []"""
        self._sample_list_split:list = []
        """List of lists representing traces. Default: []"""
        self._sample_size:int = len(self._sample)
        """Number of traces within _sample. Default: 0"""
        self._sample_event_dimension:int = 1
        """Max. number of attributes in an event. Default: 1"""
        self._sample_supported_typeset:dict = {}
        """Dict mapping support thresholds to sets ot types satisfying the corresponding support. Default: {}"""
        self._sample_att_supported_typesets:dict = {}
        """Dict mapping attributes to set of types satisfying a given support. Default: {}"""
        self._dim_sample_dict:dict = {}
        """Dictionary with dimensions as key and corresponding one-dim Sample as value."""
        self._sample_typeset = None
        self._sample_att_typesets = {}

        self._sample_vertical_sequence_database = None
        self._sample_att_vertical_sequence_database = None

        ### public
        self.sample_typeset = None
        """List of types occuring in _sample. Default: None"""
        self.sample_att_typesets = {}
        """List of types occuring in _sample. Default: None"""
        self.sample_vertical_sequence_database = None
        """Vertical map from type to occurence in trace. Default: None"""
        self.sample_att_vertical_sequence_database = None

        ### routines
        if self._sample_size:           # checks if len(given_sample) != 0
            self.set_sample_event_dimension(uniform_dimension=uniform_dimension)

    ##################################################

    
    def calc_vertical_sequence_databases(self, trace_index_list:set|None=None) -> None:
        """
            Creating the vertical database of the sample.

            Vertical means the mapping is change from '{trace number -> trace}'
            to '{att -> {type -> {trace -> [pos 0, ...]}} }' positions per
            trace. The result is storted in
            "sample_att_vertical_sequence_database".

            Meanwhile a second mapping with '{type -> {trace -> {pos -> att}}}'
            is created. The result is stored in
            "sample_vertical_sequence_database".

            Args:
                trace_index_list [= None]: optional list, which traces of
                    'sequence_database' shall be used to build the vertical
                    database. Enumeration begins with 0 and an 'traces == []'
                    includes all sequences in 'sequence_database'.

            Returns:
                None

            Raises:
                EmptySampleError: If the given sample is empty.
                IndexError: If an index occurs in trace_index_list that is not between 0 and len(sequence_database)-1
        """
        if self._sample == []:
            raise EmptySampleError("Can't create a vertical database for an empty sample!")
        if trace_index_list is None:
            trace_index_list = set(range(0,self._sample_size))
        else:
            trace_index_list = set(trace_index_list)
            for item in trace_index_list:
                if item not in range(0,len(self._sample)):
                    raise IndexError

        vertical_representation = {}
        att_vertical_representation = {att : {} for att in range(0,self._sample_event_dimension)}

        for trace_index in trace_index_list:
            events = self._sample[trace_index].split()
            for event_index, event in enumerate(events):
                values = event.split(";")
                for attribute, value in enumerate(values):
                    if value == "":
                        continue
                    if value not in att_vertical_representation[attribute]:
                        att_vertical_representation[attribute][value] = {}
                        if value not in vertical_representation:
                            vertical_representation[value] = {}
                    if trace_index not in att_vertical_representation[attribute][value]:
                        att_vertical_representation[attribute][value][trace_index] = [event_index]
                        if trace_index not in vertical_representation[value]:
                            vertical_representation[value][trace_index] = {}
                    else:
                        att_vertical_representation[attribute][value][trace_index].append(event_index)

                    if event_index not in vertical_representation[value][trace_index]:
                        vertical_representation[value][trace_index][event_index] = [attribute]
                    else:
                        vertical_representation[value][trace_index][event_index].append(attribute)

        self._sample_vertical_sequence_database = vertical_representation
        self._sample_att_vertical_sequence_database = att_vertical_representation

    def calc_sample_typeset(self, attribute:int=-1, calculate_all:bool=False) -> None:
        """
            Determines and sets _sample_typeset and _sample_att_typesets.

            Args:
                attribute [=-1]: Optional if only a specific domain shall be
                    returned. For "-1" the whole typeset is returned.

            Returns:
                None

            Raises:
                IndexError: if attribute is less than 0 or greater than or
                    equals the event dimension, or is not -1.
        """
        if calculate_all:
            if self._sample_att_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()
            att_vsdb = self._sample_att_vertical_sequence_database
            assert att_vsdb is not None

            for dim in range(0, self._sample_event_dimension):
                if dim not in self._sample_att_typesets:
                    self._sample_att_typesets[dim] = set(att_vsdb[dim].keys())

        if attribute == -1:
            if self._sample_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()

            vsdb = self._sample_vertical_sequence_database
            assert vsdb is not None
            self._sample_typeset = set(vsdb.keys())

        elif attribute in range(0, self._sample_event_dimension):
            if self._sample_att_vertical_sequence_database is None:
                self.calc_vertical_sequence_databases()
            if attribute not in self._sample_att_typesets:
                att_vsdb = self._sample_att_vertical_sequence_database
                assert att_vsdb is not None
                self._sample_att_typesets[attribute] = set(att_vsdb[attribute].keys())
        else:
            raise IndexError("Attribute must be an interger from '-1' and 'event dimension -1'")

    def calc_sample_supported_typeset(self, support:float, attribute:int=-1) -> set:
        """
            Collects all types which satisfy the given support (for the given
            attribute).

            Args:
                support: Float between 0 and 1.

                attribute: Integer which determines the attribute for which the
                    supported types should get collected. Use attribute = 0 to
                    access the first attribute. Default is -1, implying types
                    are collected across all attributes.

            Returns:
                A set consisting all types which satisfy the given support.

            Raises:
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
                IndexError: If attribute is less than 0 or greater than or equals
                    the event dimension, or is not -1.
        """
        if support < 0.0 or support > 1.0:
            raise InvalidQuerySupportError(f'Support {support} has to be between 0 and 1.')
        if attribute > self._sample_event_dimension:
            raise InvalidSampleDimensionError(f'Attribute {attribute} has to be smaller than or equal to the sample event dimension {self._sample_event_dimension}')
        sample_sized_support = ceil(self._sample_size * support)

        if attribute == -1:
            if sample_sized_support not in self._sample_supported_typeset:
                if self._sample_vertical_sequence_database is None:
                    self.calc_vertical_sequence_databases()
                vsdb = self._sample_vertical_sequence_database
                assert vsdb is not None
                supported_type_set = {symbol for symbol in vsdb if len(vsdb[symbol]) >= sample_sized_support}
                self._sample_supported_typeset[sample_sized_support] =  supported_type_set
            return self._sample_supported_typeset[sample_sized_support]
        else:
            if attribute not in range(0, self._sample_event_dimension):
                raise IndexError
            if sample_sized_support not in self._sample_att_supported_typesets:
                self._sample_att_supported_typesets[sample_sized_support] = {}
            if attribute not in self._sample_att_supported_typesets[sample_sized_support]:
                if self._sample_att_vertical_sequence_database is None:
                    self.calc_vertical_sequence_databases()
                assert self._sample_att_vertical_sequence_database is not None
                vsdb = self._sample_att_vertical_sequence_database[attribute]
                supported_type_set = {symbol for symbol in vsdb if len(vsdb[symbol]) >= sample_sized_support}
                self._sample_att_supported_typesets[sample_sized_support][attribute] =  supported_type_set
            return self._sample_att_supported_typesets[sample_sized_support][attribute]

    def calc_dim_sample_dict(self) -> None:
        """
            Calculates the one-dim Sample for each dimension 
            and sets the attribute self.dim_sample_dict.
        """
        sample_list = self._sample
        dim_count = self._sample_event_dimension
        gen_event= ';' * dim_count
        gen_event_list = [i for i in gen_event]
        dim_samples_list = []
        dim_sample_dict = {}
        for trace_id , trace in enumerate(sample_list):
            domain_list=[]
            trace_list = [domain.split(';')[:-1] for domain in trace.split()]

            for i in range(dim_count):
                current_domain=[]
                for event in trace_list:
                    cur_event_list= gen_event_list[:i] + [event[i]] + gen_event_list[i:]
                    cur_event=''.join(cur_event_list)
                    # current_domain.append(event[i] + ';')
                    current_domain.append(cur_event)
                domain_list.append(current_domain)
            dim_samples_list.append(domain_list)

        for dim in range(dim_count):
            dim_sample_list = []
            for trace_id, trace in enumerate(dim_samples_list):
                dim_sample_list.append(dim_samples_list[trace_id][dim])
            dim_sample_list = [' '.join(trace) for trace in dim_sample_list]
            dim_sample=MultidimSample()
            dim_sample.set_sample(dim_sample_list)
            dim_sample.get_sample_typeset()
            dim_sample.get_vertical_sequence_database()
            dim_sample_dict[dim] = dim_sample
        self._dim_sample_dict = dim_sample_dict

    ##################################################

    def get_sample_min_trace(self) -> tuple[str,int]:
        """
            Returns the trace with minimal number of events within the sample.

            Returns:
                A trace with minimal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')
        else:
            min_trace = self._sample[0]
            min_trace_spaces = min_trace.count(" ")
            for trace in self._sample:
                if trace.count(" ") < min_trace_spaces:
                    min_trace = trace
                    min_trace_spaces = trace.count(" ")
            return min_trace, min_trace_spaces + 1

    def get_sample_max_trace(self) -> str|None:
        """
            Returns the trace with maximal number of events within the sample.

            Returns:
                A trace with maximal length as a string.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')
        max_trace = ""
        for trace in self._sample:
            if trace.count(" ") > max_trace.count(" "):
                max_trace = trace
        return max_trace

    def get_sample_max_min_avg_trace(self) -> tuple:
        """
            Returns min and max trace as well as the average trace length.

            Returns:
                A tuple of traces and an integer where the first or second
                trace has minimal or maximal length, whereby the integer stores
                the average trace length within the sample.

            Raises:
                EmptySampleError: The given sample is empty.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        trace_length = 0
        trace_count = self._sample_size
        min_trace = self._sample[0]
        max_trace = ""
        for trace in self._sample:
            trace_length = trace_length+trace.count(" ")+1
            if trace.count(" ") >= max_trace.count(" "):
                max_trace = trace
            if trace.count(" ") < min_trace.count(" "):
                min_trace = trace
        avg_trace_length = trace_length/trace_count
        return tuple((min_trace,max_trace,avg_trace_length))

    def get_sample_typeset(self, attribute:int=-1) -> set:
        """
            Returns the typeset of the sample.
            If it is not computed yet, it will be calculated.

            Args:
                attribute [=-1]: Optional if only a specific domain shall be
                    returned. For "-1" the whole typeset is returned.

            Returns:
                set: containing all types

            Raises:
                None:

            Passes:
                IndexError: if attribute is less than 0 or greater than or
                    equals the event dimension, or is not -1.
        """
        if attribute == -1:
            if self._sample_typeset is None:
                self.calc_sample_typeset()
            assert self._sample_typeset is not None
            return self._sample_typeset
        else:
            if attribute not in self._sample_att_typesets:
                self.calc_sample_typeset(attribute)
            return self._sample_att_typesets[attribute]

    def get_sample_supported_typeset(self, support:float=1.0, attribute:int=-1) -> set:
        """
            Returns the supported typeset of the sample.
            If it is not computed yet, it will be calculated.

            Args:
                support [=1.0]: Optional if any other support is needed

                attribute [=-1]: Optional if only a specific domain shall be returned
                    For "-1" the whole typeset is returned

            Returns:
                set: all types that statisfy the support

            Raises:
                None

            Passes:
                IndexError: If attribute is less than 0 or greater than or equals
                    the event dimension, or is not -1.
                InvalidQuerySupportError: Support is less than 0 or greater
                    than 1.
        """
        sample_sized_support = ceil(self._sample_size * support)
        if attribute == -1:
            if self._sample_supported_typeset is None or sample_sized_support not in self._sample_supported_typeset:
                self.calc_sample_supported_typeset(support, attribute)
            return self._sample_supported_typeset[sample_sized_support]
        else:
            if sample_sized_support not in self._sample_att_supported_typesets or attribute not in self._sample_att_supported_typesets[sample_sized_support]:
                self.calc_sample_supported_typeset(support, attribute)
            return self._sample_att_supported_typesets[sample_sized_support][attribute]

    def set_sample(self, sample:list) -> None:
        """
            Sets _sample to sample and updates corresponding attributes.

            Args:
                sample: List of strings which represent traces.
        """
        if not isinstance(sample, list):
            raise TypeError("sample must be of type <list>!")
        self._sample = sample
        self.set_sample_size()
        self.set_sample_event_dimension()

        self._sample_supported_typeset = {}
        self._sample_att_supported_typesets = {}
        self.sample_typeset = None
        self.sample_att_typesets = {}
        self.sample_vertical_sequence_database = None
        self.sample_att_vertical_sequence_database = None

    def set_sample_size(self) -> None:
        """
            Determines and sets _sample_size for current _sample.
        """
        self._sample_size = len(self._sample)

    
    
    def set_sample_event_dimension(self,uniform_dimension:bool=False) -> None:
        """
            Determines and set the maximum event dimension.

            Args:
                uniform_dimension [=False]: Boolean which indicates whether the
                    dimension of all events is the same.
        """
        if len(self._sample) == 0:
            raise EmptySampleError('The given sample is empty.')

        dimension = -1
        if uniform_dimension is True:
            trace=self._sample[0]
            event = trace.split(' ')[0]
            dimension = event.count(";")
        else:
            for trace in self._sample:
                while len(trace)>0:
                    if trace[0] == " ":
                        trace = trace[1:]
                    else:
                        event = trace.split(' ')[0]
                        event_dimension = event.count(";")
                        if event_dimension>dimension:
                            dimension = event_dimension
                        trace = trace[len(event):]
        self._sample_event_dimension = dimension

    def calc_sample_list_split(self) -> None:
        """
            Determines and calculates _sample_list_split for current _sample.
        """

        if not self._sample_list_split:
            self._sample_list_split = [trace.split() for trace in self._sample]

    def get_sample_list_split(self) -> list:
        """
            Determines and sets _sample_list_split for current _sample.
        """

        if not self._sample_list_split:
            self.calc_sample_list_split()
        
        return self._sample_list_split
    ##################################################

    def set_sample_typeset(self, typeset) -> None:
        """
            Stores the typeset of the sample.

            Args:
                None

            Returns:
                None

            Raises:
                TypeError: if typeset is not of type <set>
        """
        if typeset is not None:
            if not isinstance(typeset,set):
                raise TypeError
        self._sample_typeset = typeset

    def get_sample_att_typesets(self) -> dict:
        """
            Returns the typesets of the sample seperated by the attributes.
            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "attribute" -> {types per dimension}

            Raises:
                None

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if not len(self._sample_att_typesets) == self._sample_event_dimension:
            for attribute in range(0, self._sample_event_dimension):
                self.calc_sample_typeset(attribute)
        return self._sample_att_typesets

    def set_sample_att_typesets(self, typesets) -> None:
        """
            Sets the typesets of the sample seperated by the attributes.

            Args:
                None

            Returns:
                None

            Raises:
                None

            Passes:
                TypeError, if "typesets" is not of type <dict>
        """
        if not isinstance(typesets,dict):
            raise TypeError("'Typesets' have to be of type <dict>")
        self._sample_vertical_sequence_database = typesets

    def get_vertical_sequence_database(self) -> dict:
        """
            Returns the vertical sequence database of the sample.

            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "type" -> [positions per traces]

            Raises:
                None

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if self._sample_vertical_sequence_database is None:
            self.calc_vertical_sequence_databases()
        assert self._sample_vertical_sequence_database is not None
        return self._sample_vertical_sequence_database

    def set_vertical_sequence_database(self, vsdb:dict) -> None:
        """
            Set _sample_vertical_sequences_database.

            Args:
                vsdb: a vertical sequence database of type <dict>

            Returns:
                None

            Raises:
                TypeError: if vsdb is not of type <dict>
        """
        if vsdb is not None:
            if not isinstance(vsdb,dict):
                raise TypeError
        self._sample_vertical_sequence_database = vsdb

    def get_att_vertical_sequence_database(self) -> dict:
        """
            Returns the vertical sequence database of the sample.

            If it is not computed yet, it will be calculated.

            Args:
                None

            Returns:
                Dict: "type" -> [positions per traces]

            Raises:
                None

            Passes:
                EmptySampleError: If the given sample is empty.

                IndexError: If an index occurs in trace_index_list that is not
                    between 0 and len(sequence_database)-1.
        """
        if self._sample_att_vertical_sequence_database is None:
            self.calc_vertical_sequence_databases()
        assert self._sample_att_vertical_sequence_database is not None
        return self._sample_att_vertical_sequence_database

    def set_att_vertical_sequence_database(self, vsdb:dict) -> None:
        """
            Set _sample_vertical_sequences_database

            Args:
                vsdb: a vertical sequence database of type <dict>

            Returns:
                None

            Raises:
                TypeError: if vsdb is not of type <dict>
        """
        if vsdb is not None:
            if not isinstance(vsdb,dict):
                raise TypeError
        self._sample_att_vertical_sequence_database = vsdb

    def get_dim_sample_dict(self) ->dict:
        """
            Returns the Dictionary containing for each dimension its one-dim Sample.

            Returns:
                dict: {dimension: one-dim Sample}
        """
        if not self._dim_sample_dict:
            self.calc_dim_sample_dict()
        return self._dim_sample_dict

    def set_dim_sample_dict(self, dim_sample_dict:dict):
        """Set _dim_sample_dict

        Args:
            dim_sample_dict (dict): Dictionary containing for each dimension its one-dim Sample
        """
        self._dim_sample_dict = dim_sample_dict


    ##################################################

    sample_typeset = property(get_sample_typeset, set_sample_typeset)
    sample_att_typesets = property(get_sample_att_typesets, set_sample_att_typesets)
    sample_vertical_sequence_database = property(get_vertical_sequence_database, set_vertical_sequence_database)
    sample_att_vertical_sequence_database = property(get_att_vertical_sequence_database, set_att_vertical_sequence_database)