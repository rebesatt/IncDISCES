#!/usr/bin/python3
""" Contains the classes for a hyperlinked tree and its vertex """
import sys
import logging
import itertools
import numpy as np
import re
from copy import copy
sys.path.append('.')
from src.query_multidim import MultidimQuery
#Logger Configuration:
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel('INFO')
FILE_HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter(LOG_FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)

class Vertex:
    """
        A vertex is a location to store information to a given query_string, namely the support behaviour, relations to next specific or next general queries and whether on
        construction all possible parent nodes have been found or not. The latter, is just an indicator, since it does not check if '$x0 $x0 $x1 $x1' exists when '$x0 $x0 $x0 $x0'
        is the current query_string.

        Attributes:
            query_string: A query string consists of types and variables. They
                are represented as blocks within query_string, separated by
                whitespaces. The beginning of a variable is marked by $, every
                unmarked block is a type.

            query_array: A query array is a list of types and variables and is
                the splitted version of a query_string.

            query_next_insert_index: Indicates on which position per dimension
                in the array is the next on which a new symbol (i.e. variable
                or type) can be inserted. It is '-1' on initialisation.

            matched_traces: A list containing all traces, on which the
                query_string has matched. It will be used to determine whether
                the support is fulfilled and to lower the computation for
                further query_strings, which have the current as prefix.
                It is 'None' on initialisation.

            child_vertices: A set of all vertices created of the given. Mostly
                vertices containing the current as prefix. Exceptions hold for
                interleavings for variables. It is an empty set on
                initialisation.

            parent_vertices: A set of all vertices of sub-query_strings of
                length |query_string|-1. In addition, if the subquery_string
                is not normal anymore, it will be normalized again
                (|query_string|-2).

            _bool_found_all_parents: On insertion of a query_string, this flag
                is set to True, if all parent vertices searched for have been
                found. If a single vertex is not found, the flag is set to
                False.

            descriptive: Indicates whether a more descriptive query was found
                or not. It holds True, if no more descriptive query was found.
                It is initialized on 'True'.
    """
    def __init__(self, query_string, query_array=None, query=None, event_dimension=-1):
        """
            Initialises a vertex.

            Args:
                query_string: the string will be set to the vertex.query_string

                query_array [=None]: An optional parameter to shorten the initialisation. If not given, the array will instantaneously be generated.

                event_dimension [=1]: An optional parameter to use query_next_insert_index for multiple dimensions.

            Returns:
                None

            Raises:
                None
        """
        self.query_string = query_string
        if query_array is None:
            if event_dimension == -1:
                self.query_array = query_string.split()
            else:
                self.query_array = [event.split(";")[:-1] for event in query_string.split()]
        else:
            self.query_array = query_array
        if event_dimension == -1:
            event_dimension = 1
        self.query = query
        self.query_next_insert_index = -1 * np.ones(event_dimension, dtype=np.int8)
        self.matched_traces = None
        self.child_vertices = set()
        self.parent_vertices = set()
        self._bool_found_all_parents = None
        self.descriptive = True

    def __str__(self):
        """
            Overrides the build-in __str__()-function and will be used on str(...) or print(...)

            Args:
                None

            Returns:
                String: the whole object is converted to a readable string.

            Raises:
                None
        """
        children = sorted(self.child_vertices, key=lambda item: (item.query_string.count(' '),item.query_string))
        parents = sorted(self.parent_vertices, key=lambda item: (item.query_string.count(' '),item.query_string))
        ret_string = "--- '" + str(self.query_string) + "'\nMatch: "
        if isinstance(self.matched_traces, range):
            ret_string += str(list(self.matched_traces)) + "\nChild: "
        else:
            ret_string += str(self.matched_traces) + "\nChild: "
        ret_string += "{"
        if len(children) >= 1:
            ret_string += "'"
            ret_string += str(children[0].query_string)
            ret_string += "'"
        for vertex in children[1:]:
            ret_string += ", '" + str(vertex.query_string) + "'"
        ret_string += "}\nParent:"
        ret_string += str(self._bool_found_all_parents) + " - {"
        if len(parents) >= 1:
            ret_string += "'"
            ret_string += str(parents[0].query_string)
            ret_string += "'"
        for vertex in parents[1:]:
            ret_string += ", '" + str(vertex.query_string) + "'"
        ret_string += "}"
        return ret_string

    def is_frequent(self, supp):
        """
            'is_frequent()' determines whether a query_string contained by a vertex is frequent on a given sample. The answer takes into account if all parent vertices have been
            found and how many traces support the given query_string.

            Args:
                sample: the given sample represents the reference a query is
                    checked against, if necessary.

                supp: Positive integer representing the number of traces needed
                    to fulfilled the support condition.

            Returns:
                True:  the sample matches the query_string.
                False: the sample can't match the query_string.
                None:  the query_string wasn't check yet

            Raises:
                InvalidQuerySupportError: If the given supp is no positive integer.
        """
        if self._bool_found_all_parents is False:
            return False

        if self.is_checked_on_sample():
            if (not isinstance(supp, int)) or supp < 1:
                raise InvalidQuerySupportError("Support must be a positive Integer!")
            if len(self.matched_traces) >= supp:
                return True
            return False
        return None

    def is_checked_on_sample(self):
        """
            Checks, if 'matched_traces' have been set or if they are still 'None'.
            Args:
                None

            Returns:
                True:  A check for this query_string has been performed.
                False: Else

            Raises:
                None
        """
        if self.matched_traces is None:
            return False
        return True


class HyperLinkedTree:
    """
        A hyperlinked tree is a spanning tree from its root to all leaves. Its vertices are connected by the child_vertices-Relation of each vertex. Nevertheless they are
        additionally hyperlinked to next level towards the root. Those hyperlinks help to link any vertex to the most vertices containing a sub-query_string with a size of
        |query_string| -1 (or |query_string| -1, if it had to be normalized).

        Attributes:
            root_vertex: The root_vertex is the root of the whole hyperlinked tree. It will be initialized with "" as query_string.

            _vertices: A dictionary containing all vertices in the hyperlinked tree. It maps each query_string to its vertex-id (object-id).

            _supp: Representing the number of traces to fulfill the support. It is given as positive integer.
    """
    def __init__(self, supp, event_dimension = -1):
        """
            Initialises a hyperlinked tree.

            Args:
                supp: Positive integer representing the number of traces needed
                    to fulfilled the support condition.

            Returns:
                None

            Raises:
                InvalidQuerySupportError: If the given supp is no positive integer.
        """
        if isinstance(supp, int) and supp >= 1:
            self._supp = supp
        if event_dimension == -1:
            self._multidim = False
        else:
            self._multidim = True
        self._root_vertex = Vertex("", event_dimension=event_dimension)
        self._root_vertex.matched_traces = range(0,supp)
        self._root_vertex._bool_found_all_parents = True
        self._vertices = {"": self._root_vertex}
        self.collision_counter = 0

    def find_vertex(self, query_string):
        """
            Find the vertex to a given query_string.

            Args:
                query_string: the string, for which a vertex is searched.

            Returns:
                vertex: returns a vertex, if the query_string exists.
                None:   else

            Raises:
                None
        """
        return self._vertices.get(query_string, None)

    def find_parent_vertices(self, current_vertex, break_when_missing_parent=False, break_when_non_matching_parent=False):
        """
            Finds all parent vertices except {x0 x0 x0 x0} => {x0 x0 x1 x1, x0 x1 x0 x1, x0 x1 x1 x0} on step more generalized

            Args:
                current_vertex: The vertex is the base point for the search

                break_when_missing_parent [= False]: If set, the search stops directly, if the first parent cannot be found.

                break_when_non_matching_parent [= False]: If set, the algorithm checks for each parent, if it already does not match. If so, the algorithm stops.

            Returns:
                parent_vertex_set: set of all parent vertices found.
                None: a break-condition is fulfilled

            Raises:
                None
        """
        if not self._multidim:
            # single dim
            query_array = current_vertex.query_array
            indices = list(range(0,len(query_array)))
            parent_vertex_set = set()
            for index in indices:
                current_slice = []
                parent_string = ""
                if query_array[index][0] == '$':                                    # if variable
                    current_element = query_array[index]

                    # remove indices from index-selection list if they follow each other continuously
                    index_prime = index + 1
                    while index_prime < len(query_array) and query_array[index] == query_array[index_prime]:
                        indices.remove(index_prime)
                        index_prime += 1

                    # remove second index from index-selection list, if only 2 of this kind exist
                    if query_array.count(current_element) == 2:
                        second_index = query_array.index(current_element, index+1)
                        if second_index in indices:
                            indices.remove(second_index)
                        current_slice = query_array[:index] + query_array[index+1:second_index] + query_array[second_index+1:]
                        parent_string = " ".join(current_slice)
                        parent_string = string_to_normalform(parent_string)[0]

                    else:
                        current_slice = query_array[:index] + query_array[index+1:]
                        parent_string = " ".join(current_slice)
                        parent_string = string_to_normalform(parent_string)[0]
                else:
                    current_slice = query_array[:index] + query_array[index+1:]
                    parent_string = " ".join(current_slice)

                parent_vertex = self._vertices.get(parent_string)
                if parent_vertex is None:
                    current_vertex._bool_found_all_parents = False
                    if break_when_missing_parent:
                        return None
                else:
                    parent_vertex_set.add(parent_vertex)
                    if break_when_non_matching_parent and not parent_vertex.is_frequent(self._supp):
                        return None
        else:
            # multidim
            current_vertex._bool_found_all_parents = True
            empty_event = [""]*len(current_vertex.query_next_insert_index)
            query_array = current_vertex.query_array
            parent_vertex_set = set()

            indices = [list(range(0,len(query_array))), list(range(0, len(empty_event)))]
            index_pairs = list(itertools.product(*indices))

            variable_look_up = {}
            iterating_index = 0
            variable_count = 0
            while iterating_index < len(index_pairs):
                (index,dim) = index_pairs[iterating_index]
                if query_array[index][dim] == "":
                    index_pairs.remove((index,dim))
                else:
                    if query_array[index][dim][0] == '$':
                        current_variable = query_array[index][dim]
                        if current_variable in variable_look_up:
                            variable_look_up[current_variable][1].append((index,dim))
                        else:
                            variable_look_up[current_variable] = (variable_count,[(index,dim)])
                            variable_count += 1
                    iterating_index += 1
            for (index,dim) in index_pairs:
                current_slice = []
                parent_string = ""
                if query_array[index][dim][0] == '$':                                    # if variable
                    current_element = query_array[index][dim]

                    # remove second index from index-selection list, if only 2 of this kind exist
                    if len(variable_look_up[current_element][1]) == 2:
                        (second_index,dim) = variable_look_up[current_element][1][1]
                        index_pairs.remove((second_index,dim))


                        new_event_1 = query_array[index][:dim] + [""] + query_array[index][dim+1:]
                        new_event_2 = query_array[second_index][:dim] + [""] + query_array[second_index][dim+1:]
                        current_slice = query_array[:index]
                        if not new_event_1 == empty_event:
                            current_slice += [new_event_1]
                        current_slice += query_array[index+1:second_index]
                        if not new_event_2 == empty_event:
                            current_slice += [new_event_2]
                        current_slice += query_array[second_index+1:]

                        if current_slice:
                            parent_string = " ".join([';'.join(event)+";" for event in current_slice])

                            replacement_dict = {"$x"+str(i):"$x"+str(i-1) for i in range(variable_look_up[current_element][0],variable_count)}

                            replacement_dict = dict((re.escape(k), v) for k, v in replacement_dict.items())
                            lambda_pattern = re.compile("|".join(replacement_dict.keys()))
                            parent_string = lambda_pattern.sub(lambda m: replacement_dict[re.escape(m.group(0))], parent_string)

                        else:
                            parent_string = ""

                    else:
                        new_event = query_array[index][:dim] + [""] + query_array[index][dim+1:]
                        if new_event == empty_event:
                            new_event = []
                        else:
                            new_event = [new_event]
                        current_slice = query_array[:index] + new_event + query_array[index+1:]

                        parent_string = " ".join([';'.join(event)+";" for event in current_slice])

                        if (index,dim) == variable_look_up[current_element][1][0]:
                            temp_var_count = variable_look_up[current_element][0] + 1
                            while temp_var_count < variable_count:
                                if variable_look_up["$x"+str(temp_var_count)][1][0] < variable_look_up[current_element][1][1]:
                                    temp_var_count += 1
                                else:
                                    break

                            replacement_dict = {"$x"+str(i):"$x"+str(i-1) for i in range(variable_look_up[current_element][0]+1,temp_var_count)}
                            replacement_dict["$x"+str(variable_look_up[current_element][0])] = "$x"+str(temp_var_count -1)

                            replacement_dict = dict((re.escape(k), v) for k, v in replacement_dict.items())
                            lambda_pattern = re.compile("|".join(replacement_dict.keys()))
                            parent_string = lambda_pattern.sub(lambda m: replacement_dict[re.escape(m.group(0))], parent_string)

                else:
                    new_event = query_array[index][:dim] + [""] + query_array[index][dim+1:]
                    if new_event == empty_event:
                        new_event = []
                    else:
                        new_event = [new_event]
                    current_slice = query_array[:index] + new_event + query_array[index+1:]
                    parent_string = " ".join([';'.join(event)+";" for event in current_slice])

                parent_vertex = self._vertices.get(parent_string)
                if parent_vertex is None:
                    current_vertex._bool_found_all_parents = False
                    if break_when_missing_parent:
                        return None
                else:
                    parent_vertex_set.add(parent_vertex)
                    if break_when_non_matching_parent and not parent_vertex.is_frequent(self._supp):
                        return None
        return parent_vertex_set

    def insert_query_string(self, existing_vertex, query_string, query_array=None, query=None, search_for_parents=True, break_when_missing_parent=False, break_when_non_matching_parent=False,
            set_descriptive_property=False):
        """
            Insert an new query_string in existent hyperlinked tree on a specific vertex. The new vertex occurs in exactly one child_vertices-list. Moreover for all sub-queries
            with one element less (and normalized - therefore maybe 2 elements less) is a check performed, if the sub_query exists in the hyperlinked tree. If so, they will be
            added to parent_vertices list.

            Args:
                existing_vertex: the vertex to which the query_string will be
                    connected to.

                query_string: the query_string which should be inserted into
                    the hyperlinked tree.

                query_array [= None]: Optional parameter to directly set the
                    query_array instead of generating it.

                search_for_parents [= True]: Optional parameter to decide,
                    whether the parents of newly inserted string should be
                    connected in a linked list.

                break_when_missing_parent [= False]: Optional parameter to
                    force a break, if one parent can't be found.

                break_when_non_matching_parent [= False]: Optional parameter to
                    force a break, if on insertion a parent is found, which
                    already is not matched be a sample.

                set_descriptive_property [=False]: Optional parameter to decide
                    whether the descriptive property of the found parents
                    shall be updated or not.

            Returns:
                Vertex: returns the newly created vertex

            Raises:
                ValueError: If the existing_vertex is not of type 'Vertex'
        """
        ex_vertex = self.find_vertex(query_string)
        if ex_vertex:
            self.collision_counter += 1
            raise NameError("Query '%s' already exists", query_string)
        if not isinstance(existing_vertex, Vertex):
            raise ValueError("'Existing_vertex' must be of type 'Vertex'!")
        if self._multidim:
            new_vertex = Vertex(query_string, query_array, query, event_dimension=len(existing_vertex.query_next_insert_index))
        else:
            new_vertex = Vertex(query_string, query_array, query)
        existing_vertex.child_vertices.add(new_vertex)
        if set_descriptive_property:
            existing_vertex.descriptive = False
        self._vertices[query_string] = new_vertex

        if search_for_parents:
            parent_vertices_set = self.find_parent_vertices(new_vertex, break_when_missing_parent, break_when_non_matching_parent)
            if parent_vertices_set:
                new_vertex.parent_vertices = parent_vertices_set
                if set_descriptive_property:
                    for vertex in parent_vertices_set:
                        vertex.descriptive = False
                if new_vertex._bool_found_all_parents is None:
                    new_vertex._bool_found_all_parents = True
        return new_vertex

    def get_root(self):
        """
            Returns the vertex representing the root.

            Args:
                None

            Returns:
                vertex: returns a root

            Raises:
                None
        """
        return self._root_vertex

    def set_match_results(self, vertex, matching_traces):
        """
            Sets the set of traces as matching result. Can additionally handle booleans as true is set to match the traces [0,...,'supp'-1] and false is set to '[]'

            Args:
                vertex: a vertex which the results are stored to.

                matching_traces: a list of trace numbers, on which the query matches.

            Returns:
                None

            Raises:
                TypeError: if 'vertex' is not of type 'Vertex'
                TypeError: if 'matching_traces' contains anything but non-negative integer.
                ValueError: if 'matching_traces' contains negative integer!")
        """
        if not isinstance(vertex, Vertex):
            raise TypeError("'vertex' must be of type 'Vertex'!")
        if isinstance(matching_traces, bool):
            if matching_traces:
                vertex.matched_traces = list(range(0,self._supp))
            else:
                vertex.matched_traces = []
        else:
            if not isinstance(matching_traces,list):
                raise TypeError("'matching_traces' must be of type list!")
            if all(isinstance(item, int) and item >= 0 for item in matching_traces):
                vertex.matched_traces = list(matching_traces)
            else:
                raise ValueError("'matching_traces' should contain only non-negative integer!")

    def vertices_to_list(self, frequent_items_only=False):
        """
            Returns all vertices in a list.

            Args:
                None

            Returns:
                list: returns a list of query_strings contained by the hyperlinked tree.

            Raises:
                None
        """
        if not frequent_items_only:
            return list(self._vertices.values())
        return [item for item in self._vertices.values() if item.is_frequent(self._supp)]

    def query_strings_to_list(self, frequent_items_only=False):
        """
            Returns all strings of vertices in a list.

            Args:
                None

            Returns:
                list: returns a list of query_strings contained by the hyperlinked tree.

            Raises:
                None
        """
        if not frequent_items_only:
            return list(self._vertices.keys())
        return [item for item in self._vertices if item.is_frequent(self._supp)]

    def query_strings_to_set(self, frequent_items_only=False):
        """
            Returns all strings of vertices in a list.

            Args:
                None

            Returns:
                set: returns a set of query_strings contained by the hyperlinked tree.

            Raises:
                None
        """
        if not frequent_items_only:
            return set(self._vertices.keys())
        return {string for (string,vertex) in self._vertices.items() if vertex.is_frequent(self._supp)}

    def __str__(self):
        """
            Overrides the build-in __str__()-function and will be used on str(...) or print(...)

            Args:
                None

            Returns:
                String: the whole object is converted to a readable string.

            Raises:
                None
        """
        ret_string = str(list(self._vertices.items())[0][1])
        for key_value_pair in list(self._vertices.items())[1:]:
            ret_string += "\n"
            ret_string += str(key_value_pair[1])
        return ret_string


def string_to_normalform(query_string:str, repeated_variables:set|None=None, query_windowsize_local:list|None=None) -> tuple:
    """
        Converts the query_string into normalform.

        A query string with neither gap constraints nor a global window size is
        in normalform iff it does not contain non-repreated variables. This
        function deletes all non-repeated variables and merges the local window
        sizes.

        Args:
            query_string: a query_string.

            repeated_variables [= None]: optional parameter, if the
                repeated_variables are already known

            query_windowsize_local [= None]: optional parameter, if the
                query_string contains local windowsizes. They will be adjusted
                on normalization process.

        Returns:
            tuple of query_string, query_windowsize_local.
                They contain the updated versions.

        Raises:
            None
    """
    LOGGER.debug('string_to_normalform - Started')
    if repeated_variables is None:
        # find repeated variables
        repeated_variable_dict = {}
        repeated_variables = set()

        query_array = query_string.split(' ')
        for item in query_array:
            if len(item) > 0 and item[0] == '$':
                if item in repeated_variable_dict:
                    repeated_variable_dict[item] += 1
                else:
                    repeated_variable_dict[item] = 1

        for item, value in repeated_variable_dict.items():
            if value > 1:
                repeated_variables.add(item[1:])

    query_string_nf = copy(query_string)
    query_string_nf_prefix = ""
    pos = 0
    while True:
        if pos >= len(query_string_nf) -1:
            break
        if query_string_nf[pos] == "$":
            variable = query_string_nf[pos+1:]
            variable = variable.split(' ')[0]
            LOGGER.debug('string_to_normalform - Current variable %s', variable)
            if variable not in repeated_variables:
                LOGGER.debug('string_to_normalform - Current variable %s is not repeated and has to be deleted', variable)
                # Delete the repeated variable in the query string
                if pos == 0:
                    query_string_nf = query_string_nf[len(variable)+2:]
                    query_string_nf_string_length = query_string_nf.count(" ")+1
                else:
                    query_string_nf_prefix = query_string_nf[0:pos-1]
                    query_string_nf_suffix = query_string_nf[pos+len(variable)+1:]
                    query_string_nf = query_string_nf_prefix + query_string_nf_suffix
                    query_string_nf_string_length = query_string_nf.count(" ")+1
                pos = pos-1

                # Update the local window sizes
                if query_windowsize_local:
                    LOGGER.debug('string_to_normalform - Local window size tuples have to be updated')
                    if pos <= 0:
                        tuple_index = -1
                        window_size_tuple_at_first_index = list(query_windowsize_local[0])
                        window_size_tuple_at_second_index = list(query_windowsize_local[1])
                    else:
                        tuple_index = query_string_nf_prefix.count(" ")+1
                        window_size_tuple_at_first_index = list(query_windowsize_local[tuple_index])
                        window_size_tuple_at_second_index = list(query_windowsize_local[tuple_index+1])

                    # Variable at first position in query string is deleted
                    if tuple_index == -1:
                        LOGGER.debug('string_to_normalform - Update the first local window size tuple')
                        # Variable is also the last block within the query string
                        if len(query_string_nf)==0:
                            if window_size_tuple_at_second_index[0] == -1:
                                new_lower_bound = window_size_tuple_at_first_index[0] + 1
                                new_upper_bound = window_size_tuple_at_first_index[1] + 1
                            else:
                                new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                                new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                        else:
                            if window_size_tuple_at_first_index[0] == -1:
                                new_lower_bound = window_size_tuple_at_second_index[0] + 1
                                new_upper_bound = window_size_tuple_at_second_index[1] + 1
                            else:
                                new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                                new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                    # Variable at last position in query string is deleted
                    elif query_string_nf_string_length==tuple_index:
                        LOGGER.debug('string_to_normalform - Update the last local window size tuple')
                        if window_size_tuple_at_second_index[0] == -1:
                            new_lower_bound = window_size_tuple_at_first_index[0] + 1
                            new_upper_bound = window_size_tuple_at_first_index[1] + 1
                        else:
                            new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                            new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                    else:
                        new_lower_bound = window_size_tuple_at_first_index[0] + window_size_tuple_at_second_index[0] + 1
                        new_upper_bound = window_size_tuple_at_first_index[1] + window_size_tuple_at_second_index[1] + 1
                    window_size_tuple = (new_lower_bound, new_upper_bound)

                    LOGGER.debug('string_to_normalform - Build new list of local window size tuples')
                    updated_windowsize_local = []
                    count = 0
                    while len(query_windowsize_local) > 0:
                        if count == 0 and tuple_index == -1:
                            LOGGER.debug('string_to_normalform - Add new first local window size tuple')
                            updated_windowsize_local.append(window_size_tuple)
                            query_windowsize_local.pop(0)
                            query_windowsize_local.pop(0)
                            count = count+2
                        elif count == tuple_index:
                            LOGGER.debug('string_to_normalform - Add new local window size tuple')
                            updated_windowsize_local.append(window_size_tuple)
                            query_windowsize_local.pop(0)
                            query_windowsize_local.pop(0)
                            count = count+2
                        else:
                            LOGGER.debug('string_to_normalform - Add unchanged local window size tuple')
                            updated_windowsize_local.append(query_windowsize_local[0])
                            query_windowsize_local.pop(0)
                            count = count+1
                    query_windowsize_local = updated_windowsize_local

                LOGGER.debug('string_to_normalform - Current variable %s is now deleted', variable)
            else:
                pos = pos+len(variable)
        pos = pos + 1

    # Variables are renamed in order of first appearance to $x0,$x1 and so on
    gen_variables=list(dict.fromkeys([s for s in query_string_nf.split() if "$" in s]))
    vars_nf= ['$x' + str(i) for i in range(len(gen_variables))]
    new_vars = { v:k for k,v in dict(zip( vars_nf, gen_variables)).items()}
    query_string_nf=" ".join([new_vars.get(item,item)  for item in query_string_nf.split()])

    LOGGER.debug('string_to_normalform - Query string in normalform: %s', query_string)
    LOGGER.debug('string_to_normalform - Finished')

    return query_string_nf, query_windowsize_local
