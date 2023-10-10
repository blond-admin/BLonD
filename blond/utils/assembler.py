# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the implementation of the Assembler class.

:Authors: **Konstantinos Iliakis**

"""


from __future__ import annotations
import importlib
import inspect
import logging
from blond.utils.track_iteration import TrackIteration
from blond.utils import timing
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)

# Need to define an abstract class Trackable
# that will have an abstract method track


class Trackable(ABC):
    '''
    Abstract class for trackable elements
    '''

    def __init__(self, period: int = 1, priority: 'Optional[int]' = None) -> None:
        '''Constructor

        Args:
            period (int, optional): Period in turn in between calls to the tracking method. Defaults to 1.
            priority (Optional[int], optional): Tracking priority. Higher values will be tracked first. Defaults to None.
        '''
        self.period = period
        self.priority = priority

    @abstractmethod
    def track(self) -> None:
        '''Abstract track method. Has to be implemented by the user.
        '''
        pass


class Assembler:
    '''
    Assembler class
    '''

    '''
    This is the default tracking priority of blond trackable elements.
    Priority 0 is reserved for custom elements.
    Higher priority number means that the element will be tracked first.
    '''
    tracking_priority_dict = {'Profile': 800,
                              '_InducedVoltage': 700,
                              'TotalInducedVoltage': 700,
                              'SynchrotronRadiation': 600,
                              'BeamFeedback': 500,
                              'SPSCavityFeedback': 500,
                              'CavityFeedback': 500,
                              'LHCNoiseFB': 400,
                              'RingAndRFTracker': 300,
                              'FullRingAndRF': 200,
                              'Plot': 100,
                              'BunchMonitor': 100,
                              'SlicesMonitor': 100,
                              'MultiBunchMonitor': 100,
                              'default': 0
                              }


    class PipelineElement:
        '''
        PipelineElement class. Used to hold all neccaessary information for a trackable element. 
        '''

        def __init__(self, element, idx=0) -> None:
            '''Initialize PipelineElement

            Args:
                element (needs to have a track method): _description_
                idx (int, optional): The idx used when having multiple elements of the class. Defaults to 0.
            '''
            # Extract name for class
            self.class_name = element.__class__.__name__
            self.name = f'{self.class_name}-{idx}'
            self.element = element

            # Add the tracking period. If not given, assume 1 (i.e. every turn)
            self.period = getattr(element, 'period', 1)

            # Assign the tracking priority. If not specified, get it from the Assembler method
            self.priority = Assembler.get_tracking_priority(element)

            # Replace track method with the element's track method
            self.track = element.track

        def track(self):
            '''Placeholder track method, will be overwritten during initialization of the object
            '''
            raise NotImplementedError("Track method not implemented")


    @staticmethod
    def get_tracking_priority(element: Trackable) -> int:
        '''Get tracking priority of a Trackable element

        Args:
            element (Trackable): If not instance of Trackable, then it is a blond class with a track method

        Returns:
            int: The tracking priority of the element. Higher priority means that the element will be tracked first.
        '''
        class_name = element.__class__.__name__
        parent_classes = [
            parent_class.__name__ for parent_class in element.__class__.__bases__]

        if hasattr(element, 'priority'):
            # If tracking_priority is defined, return it
            return element.priority
        elif class_name in Assembler.tracking_priority_dict:
            # If class name is in tracking order, return its index
            return Assembler.tracking_priority_dict[class_name]
        elif any(parent_class in Assembler.tracking_priority_dict for parent_class in parent_classes):
            #  If any of the parent_classes is in tracking order, return its index
            for parent_class in parent_classes:
                if parent_class in Assembler.tracking_priority_dict:
                    return Assembler.tracking_priority_dict[parent_class]
        else:
            # Else we have a custom class, return the default priority.
            return Assembler.tracking_priority_dict['default']


    @staticmethod
    def sort_pipeline(pipeline: 'List[PipelineElement]') -> 'List[PipelineElement]':
        '''Sort pipeline according to tracking order

        Args:
            pipeline (List[Callable]): _description_

        Returns:
            List[Callable]: _description_
        '''
        # Sort according to tracking order
        pipeline = sorted(
            pipeline, key=lambda x: x.priority, reverse=True)
        return pipeline


    @staticmethod
    def is_trackable(elem) -> bool:
        '''_summary_

        Args:
            elem (_type_): _description_

        Returns:
            bool: _description_
        '''
        return issubclass(type(elem), Trackable) or (hasattr(elem, 'track') and callable(elem.track))


    @staticmethod
    def is_callable_tuple(elem) -> bool:
        '''_summary_

        Args:
            elem (_type_): _description_

        Returns:
            bool: _description_
        '''
        # Check if elem is a tuple in the form (func, (func_args))
        return isinstance(elem, tuple) and len(elem) == 2 and callable(elem[0]) and isinstance(elem[1], tuple)


    @staticmethod
    def split_args_kwargs(arguments: 'Union[Tuple, Dict]') -> 'Tuple[Tuple, Dict]':
        '''
        Split arguments into positional and keyword arguments
        Arguments can be in the form:
        - dictionary
        - tuple in the form (positional_args, keyword_args)
        - tuple in the form (positional_args)
        Warning! If only positional arguments are given, and the last argument is a dictionary,
        it will be interpreted as keyword arguments
        '''
        if isinstance(arguments, dict):
            return (), arguments
        elif isinstance(arguments, tuple) and len(arguments) == 2 and isinstance(arguments[0], tuple) and isinstance(arguments[1], dict):
            return arguments[0], arguments[1]
        elif isinstance(arguments, tuple):
            return arguments, {}


    @staticmethod
    def discover_blond_distributions() -> 'Dict':
        '''
        Discover all blond distributions
        '''

        modules = ['blond.beam.distributions',
                   'blond.beam.distributions_multibunch', 'blond.toolbox.tomoscope']

        blond_distributions = {}
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                logger.debug(f'Error importing module {module_name}: {e}')
                continue

            for name, member in inspect.getmembers(module):
                if inspect.isfunction(member) and not name.startswith('_'):
                    blond_distributions[name] = member

        return blond_distributions


    @staticmethod
    def discover_blond_classes() -> 'Dict[str, Callable]':
        '''
        Discover all blond classes
        '''

        modules = [
            'blond.beam.beam',
            'blond.beam.coasting_beam',
            'blond.beam.distributions_multibunch',
            'blond.beam.distributions',
            'blond.beam.profile',
            'blond.beam.sparse_slices',
            'blond.impedances.impedance',
            'blond.impedances.impedance_sources',
            'blond.impedances.induced_voltage_analytical',
            'blond.impedances.music',
            'blond.input_parameters.rf_parameters_options',
            'blond.input_parameters.rf_parameters',
            'blond.input_parameters.ring_options',
            'blond.input_parameters.ring',
            'blond.llrf.beam_feedback',
            'blond.llrf.cavity_feedback',
            'blond.llrf.impulse_response',
            'blond.llrf.notch_filter',
            'blond.llrf.offset_frequency',
            'blond.llrf.rf_modulation',
            'blond.llrf.rf_noise',
            'blond.llrf.signal_processing',
            'blond.monitors.monitors',
            'blond.plots.plot_beams',
            'blond.plots.plot_impedance',
            'blond.plots.plot_llrf',
            'blond.plots.plot_parameters',
            'blond.plots.plot',
            'blond.plots.plot_slices',
            'blond.synchrotron_radiation.synchrotron_radiation',
            'blond.toolbox.tomoscope',
            'blond.trackers.tracker',
            'blond.trackers.utilities',
        ]

        blond_classes = {}
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                logger.debug(f'Error importing module {module_name}: {e}')
                continue

            for name, member in inspect.getmembers(module):
                if inspect.isclass(member) and not name.startswith('_'):
                    blond_classes[name] = member

        return blond_classes


    def __init__(self, element_list=[]) -> None:
        '''
        Initialize assembler

        Args:
            element_list (list): list of elements to be tracked

        Attributes:
            objects (Dict): dictionary of all blond objects found
            pipeline (List): list of PipelineElement trackable elements
            pipeline_tracker (TrackIteration): track iteration object
            element_list (List): list of elements as provided by the user
            is_built (bool): flag to indicate if the assembler has been built
        '''

        self.active_objects = {}
        self.pipeline = []
        self.pipeline_tracker = None
        self.element_list = element_list
        self.__is_built = False
        self.element_idx = 0
        self.with_timing = False

        # Discover the blond distributions
        self.blond_distributions = Assembler.discover_blond_distributions()
        # Discover all blond classes
        self.blond_classes = Assembler.discover_blond_classes()
        # Reset timing
        timing.reset()

    @property
    def is_built(self) -> bool:
        return self.__is_built
    
    @is_built.setter
    def is_built(self, value: bool) -> None:
        if self.__is_built and not value:
            # Unbuild the pipeline
            self.remove_properties()
            self.active_objects = {}
            self.pipeline = []
            self.pipeline_tracker = None

            # logger.debug('Assembler is already built. Call build method to rebuild.')
        self.__is_built = value


    def __str__(self) -> str:
        '''_summary_

        Returns:
            str: _description_
        '''
        string = f'Assembler with {len(self.pipeline)} tracking elements and {len(self.active_objects)} objects.'
        string += '\nPipeline elements:'
        for i, elem in enumerate(self.pipeline):
            string += f'\n\t{i}. {elem.name}'

        string += '\nObjects:'
        for key in self.active_objects.keys():
            string += f'\n\t{key}'

        return string


    def init_object_from_dict(self, elem_class, elem_args, elem_kwargs):
        '''_summary_

        Args:
            elem_class (_type_): _description_
            elem_args (_type_): _description_
            elem_kwargs (_type_): _description_

        Returns:
            _type_: _description_
        '''
        return self.blond_classes[elem_class](*elem_args, **elem_kwargs)


    def promote_to_properties(self, elem_dict) -> None:
        '''_summary_

        Args:
            elem_dict (_type_): _description_
        '''

        # delete old properties
        self.remove_properties()

        # add the new objects to the self.__dict__
        self.__dict__.update(elem_dict)


    def remove_properties(self) -> None:
        '''Remove all the active_objects from the assembler's properties
        '''

        # delete all old references from self.__dict__
        for key in self.active_objects.keys():
            if key in self.__dict__:
                del self.__dict__[key]


    def replace_object_references(self, args, kwargs) -> 'Tuple[Tuple, Dict]':
        '''Check for values that are pointing to previously initialized objects
        and replace them with references to the initialized objects

        Args:
            args (_type_): _description_
            kwargs (_type_): _description_

        Returns:
            Tuple[Tuple, Dict]: _description_
        '''

        # First check kwargs
        for k, v in kwargs.items():
            if isinstance(v, str) and v in self.active_objects:
                kwargs[k] = self.active_objects[v]
        # Then check args. Convert to list to support assignment
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg in self.active_objects:
                args[i] = self.active_objects[arg]
        return tuple(args), kwargs


    def convert_from_dictionary_to_object(self, elem: 'Dict') -> 'Any':
        '''_summary_

        Args:
            elem (Dict): _description_
        '''
        assert len(
            elem) == 1, 'Dictionary elements must be in the form: {classname: {arg1: val1, arg2:val2, ...}}'
        elem_class, (elem_all_args) = next(iter(elem.items()))
        elem_args, elem_kwargs = Assembler.split_args_kwargs(elem_all_args)
        logger.debug(
            f'Found dictionary element with {len(elem_args)} positional arguments and {len(elem_kwargs)} kwargs')
        # Replace reference kwargs objects
        elem_args, elem_kwargs = self.replace_object_references(
            elem_args, elem_kwargs)
        elem = self.init_object_from_dict(elem_class, elem_args, elem_kwargs)

        return elem


    def build_pipeline(self) -> None:
        '''Build the pipeline from the element list.
        Elements can be either objects or dictionaries.
        '''

        new_active_objects = {}
        for elem in self.element_list:
            if isinstance(elem, dict):
                self.convert_from_dictionary_to_object(elem)

            elem_class = type(elem).__name__
            logger.debug(f'Found element of type: {elem_class}')

            if Assembler.is_trackable(elem):
                # if trackable, need to add its track method to the pipeline
                self.pipeline.append(Assembler.PipelineElement(
                    elem, idx=self.element_idx))

            # Since the assembler also has a record of all objects, need to store object in correct attribute
            if elem_class in new_active_objects:
                if not isinstance(new_active_objects[elem_class], list):
                    new_active_objects[elem_class] = list(new_active_objects[elem_class])
                new_active_objects[elem_class].append(elem)
            else:
                new_active_objects[elem_class] = elem

            self.element_idx += 1

        # Promote the new objects to properties
        self.promote_to_properties(new_active_objects)
        # Update the active objects
        self.active_objects = new_active_objects

        # Sort according to custom order
        self.pipeline = Assembler.sort_pipeline(self.pipeline)
        # Extract tracking periods
        tracking_periods = [elem.period for elem in self.pipeline]
        # extract the tracking methods
        track_methods = [elem.track for elem in self.pipeline]

        self.pipeline_tracker = TrackIteration(
            track_methods, initTurn=0, finalTurn=-1, trackPeriods=tracking_periods)
        self.is_built = True


    def build_distribution(self, distribution_dict: 'Optional[Dict]' = None,
                           distribution_func: 'Optional[Callable]' = None,
                           distribution_args: 'Tuple' = ()) -> None:
        '''Used to build a blond distribution from a dictionary or a function. 
        Either the distribution_dict or the distribution_func must be provided.

        Args:
            distribution_dict (Optional[Dict]): Dictionary in the form {distribution_type: {arg1: val1, arg2:val2, ...}}.
                Defaults to None.
            distribution_func (Optional[Callable]): Distribution function. Defaults to None.
            distribution_args (Tuple, optional): Distribution arguments. Defaults to ().
        '''
        if distribution_dict is not None and isinstance(distribution_dict, dict):
            # If distribution is a dictionary, then extract distribution type and arguments
            assert len(
                distribution_dict) == 1, 'Dictionary elements must be in the form: {disrtibution_type: {arg1: val1, arg2:val2, ...}}'
            distr_type, (distribution_args) = next(
                iter(distribution_dict.items()))

            assert distr_type in self.blond_distributions, f'Distribution type not recognized: {distr_type}'
            # extract the distribution function
            distribution_func = self.blond_distributions[distr_type]

        assert callable(
            distribution_func), 'Distribution function not provided, or not callable.'

        # Split arguments into positional and keyword arguments
        distr_args, distr_kwargs = Assembler.split_args_kwargs(
            distribution_args)

        # Replace references to blond objects
        distr_args, distr_kwargs = self.replace_object_references(
            distr_args, distr_kwargs)

        # Call the distribution function
        distribution_func(*distr_args, **distr_kwargs)


    def track(self, num_turns: int = 1, with_timing: bool = False) -> None:
        '''Track all trackable pipeline objects for a number of turns

        Args:
            num_turns (int, optional): _description_. Defaults to 1.
            with_timing (bool, optional): _description_. Defaults to False.
        '''
        if not self.is_built:
            print('Warning: Object not built. Call build method first.')
            return

        if with_timing and not self.with_timing:
            # wrap all elements in the pipeline with timing
            pipeline = [timing.timeit(key=stage.name)(stage.track)
                        for stage in self.pipeline]
            self.pipeline_tracker._map = pipeline
        elif not with_timing and self.with_timing:
            # Remove timing from all pipeline elements
            pipeline = [stage.track for stage in self.pipeline]
            self.pipeline_tracker._map = pipeline

        self.with_timing = with_timing

        # Track all objects for the given number of turns
        self.pipeline_tracker(num_turns)


    def __insert_at(self, element: 'Any', index: int = 0) -> None:
        '''Insert element at index, shifting all elements at and after the index to the right

        Args:
            element (_type_): Insert element in the list of BLonD elements.
            index (int, optional): Position to insert element. Defaults to 0.
        '''
        self.element_list.insert(index, element)
        self.is_built = False


    def append(self, element: 'Any') -> None:
        '''Append element to the end of the element list.

        Args:
            element (_type_): _description_
        '''
        self.__insert_at(element, len(self.element_list))


    def insert(self, element: 'Any') -> None:
        '''Synonym for append.
        '''
        self.append(element)


    def remove_last(self) -> None:
        '''Remove last element from the element list.
        '''
        self.remove(self.element_list[-1])
        self.is_built = False


    def report_timing(self) -> None:
        '''Report timing information after tracking.
        '''
        timing.report()


    def to_yaml(self) -> None:
        '''Convert pipeline to yaml file.
        '''
        pass


    @classmethod
    def from_yaml(cls, file_name) -> Assembler:
        '''Read yaml pipeline elements from yaml file

        Args:
            file_name (_type_): _description_

        Returns:
            Assembler: _description_
        '''
        import yaml

        # convert from yaml to dict
        with open(file_name, 'r') as file:
            element_dict = yaml.safe_load(file)

        # then call init method
        return cls(element_dict)


# THese are methods that will be probably deleted in the future. 
# def __prepend(self, element: 'Any') -> None:
#     '''Prepend element to the beginning of the element list

#     Args:
#         element (_type_): _description_
#     '''
#     self.__insert_at(element, 0)

# def remove(self, element) -> None:
#     '''Remove element from the element list

#     Args:
#         element (_type_): _description_
#     '''
#     index = self.element_list.index(element)
#     self.is_built = False

# @staticmethod
# def get_function_name(func: 'Callable') -> str:
#     '''_summary_

#     Args:
#         func (Callable): _description_

#     Returns:
#         str: _description_
#     '''
#     return func.__func__.__globals__['__name__']
# @staticmethod
# def is_dictionary(elem) -> bool:
#     '''_summary_

#     Args:
#         elem (_type_): _description_

#     Returns:
#         bool: _description_
#     '''
#     return isinstance(elem, dict)