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
    def __init__(self, period: int =1) -> None:
        self.period = period
        # self.place_after = None
        # self.place_before = None

    @abstractmethod
    def track(self) -> None:
        '''Abstract track method. Has to be implemented by the user.
        '''
        pass

class PipelineElement(Trackable):
    '''
    PipelineElement class
    '''
    def __init__(self, element, period: int =1) -> None:
        super().__init__(period)
        self.name = element.__class__.__name__
        self.element = element
        self.track = element.track


class Assembler:
    '''
    Assembler class
    '''

    # This is the default tracking order of blond trackable elements
    tracking_order = ['Profile',
                      '_InducedVoltage', 'TotalInducedVoltage',
                       'SynchrotronRadiation',
                       'BeamFeedback',
                       'SPSCavityFeedback', 'CavityFeedback',
                       'LHCNoiseFB',
                       'RingAndRFTracker', 'FullRingAndRF',
                       'Plot',
                       'BunchMonitor', 'SlicesMonitor', 'MultiBunchMonitor',
                       ]

    @staticmethod
    def get_tracking_order_idx(obj: 'Any') -> int:
        '''Get tracking order index of a function

        Args:
            Obj (Callable): _description_

        Returns:
            int: _description_
        '''
        class_name = obj.__self__.__class__.__name__
        parent_classes = [parent_class.__name__ for parent_class in obj.__class__.__bases__]

        if class_name in Assembler.tracking_order:
            # If class name is in tracking order, return its index
            return Assembler.tracking_order.index(class_name)
        elif any(parent_class in Assembler.tracking_order for parent_class in parent_classes):
            #  If any of the parent_classes is in tracking order, return its index
            for parent_class in parent_classes:
                if parent_class in Assembler.tracking_order:
                    return Assembler.tracking_order.index(parent_class)
        else:
            # Else we have a custom class, place it at the end. 
            return len(Assembler.tracking_order)


    @staticmethod
    def sort_pipeline(pipeline: 'List[Callable]') -> 'List[Callable]':
        '''Sort pipeline according to tracking order

        Args:
            pipeline (List[Callable]): _description_

        Returns:
            List[Callable]: _description_
        '''
        # Sort according to tracking order
        pipeline = sorted(pipeline, key=lambda x: Assembler.get_tracking_order_idx(x))
        return pipeline


    @staticmethod
    def get_function_name(func: 'Callable') -> str:
        '''_summary_

        Args:
            func (Callable): _description_

        Returns:
            str: _description_
        '''
        return func.__func__.__globals__['__name__']

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
    def is_dictionary(elem)-> bool:
        '''_summary_

        Args:
            elem (_type_): _description_

        Returns:
            bool: _description_
        '''
        return isinstance(elem, dict)
    
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
        if Assembler.is_dictionary(arguments):
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
        
        modules = ['blond.beam.distributions', 'blond.beam.distributions_multibunch']

        blond_distributions = {}
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                logger.debug(f'Error importing module {module_name}: {e}')
                continue

            for name, member in inspect.getmembers(module):
                if inspect.isfunction(member) and not name.startswith('_') :
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
            'blond.gpu.butils_wrap_cupy',
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
            'blond.toolbox.action',
            'blond.toolbox.diffusion',
            'blond.toolbox.filters_and_fitting',
            'blond.toolbox.logger',
            'blond.toolbox.next_regular',
            'blond.toolbox.parameter_scaling',
            'blond.toolbox.tomoscope',
            'blond.trackers.tracker',
            'blond.trackers.utilities',
            'blond.utils.bmath',
            'blond.utils.butils_wrap_cpp',
            'blond.utils.butils_wrap_python',
            'blond.utils.data_check',
            'blond.utils.exceptions',
            'blond.utils.mpi_config',
            'blond.utils.timing',
            'blond.utils.track_iteration',
        ]

        blond_classes = {}
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                logger.debug(f'Error importing module {module_name}: {e}')
                continue
            
            for name, member in inspect.getmembers(module):
                if inspect.isclass(member) and not name.startswith('_') :
                    blond_classes[name] = member
        
        return blond_classes


    def __init__(self, element_list=[]) -> None:
        '''
        Initialize assembler

        Args:
            element_list (list): list of elements to be tracked
        '''

        self.objects = {}
        self.pipeline = []
        self.pipeline_tracker = None
        self.element_list = element_list
        self.tracking_periods = []

        self.active_objects = {}
        self.is_built=False

        self.blond_distributions = Assembler.discover_blond_distributions()
        self.blond_classes = Assembler.discover_blond_classes()
        # Reset timing
        timing.reset()

    def __str__(self) -> str:
        '''_summary_

        Returns:
            str: _description_
        '''
        string = f'Assembler with {len(self.pipeline)} tracking elements and {len(self.objects)} objects.'
        string += '\nPipeline elements:'
        for i, elem in enumerate(self.pipeline):
            string += f'\n\t{i}. {Assembler.get_function_name(elem)}'

        string += '\nObjects:'
        for key in self.objects.keys():
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


    def promote_to_attributes(self, elem_dict) -> None:
        '''_summary_

        Args:
            elem_dict (_type_): _description_
        '''
        
        # delete all old references from self.__dict__
        for key in self.active_objects.keys():
            if key in self.__dict__:
                del self.__dict__[key]
        
        # add the new objects to the self.__dict__
        self.__dict__.update(elem_dict)
        self.active_objects = elem_dict
    
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
            if isinstance(v, str) and v in self.objects:
                kwargs[k] = self.objects[v]
        # Then check args. Convert to list to support assignment
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg in self.objects:
                args[i] = self.objects[arg]
        return tuple(args), kwargs

    def convert_from_dictionary_to_object(self, elem: 'Dict') -> 'Any':
        '''_summary_

        Args:
            elem (Dict): _description_
        '''
        assert len(elem) == 1, 'Dictionary elements must be in the form: {classname: {arg1: val1, arg2:val2, ...}}'
        elem_class, (elem_all_args) = next(iter(elem.items()))
        elem_args, elem_kwargs = Assembler.split_args_kwargs(elem_all_args)
        logger.debug(f'Found dictionary element with {len(elem_args)} positional arguments and {len(elem_kwargs)} kwargs')
        # Replace reference kwargs objects
        elem_args, elem_kwargs = self.replace_object_references(elem_args, elem_kwargs)
        elem = self.init_object_from_dict(elem_class, elem_args, elem_kwargs)

        return elem


    def build_pipeline(self) -> None:
        '''Build the pipeline from the element list.
        Elements can be either objects or dictionaries.
        '''

        for elem in self.element_list:
            if Assembler.is_dictionary(elem):
                self.convert_from_dictionary_to_object(elem)
            
            elem_class = type(elem).__name__
            logger.debug(f'Found element of type: {elem_class}')
            
            if Assembler.is_trackable(elem):
                # if trackable, need to add its track method to the pipeline
                self.pipeline.append(elem.track)
                # Add the tracking period. If not given, assume 1 (i.e. every turn)
                self.tracking_periods.append(getattr(elem, 'period', 1))
            
            # Since the assembler also has a record of all objects, need to store object in correct attribute 
            if elem_class in self.objects:
                if not isinstance(self.objects[elem_class], list):
                    self.objects[elem_class] = list(self.objects[elem_class])
                self.objects[elem_class].append(elem)
            else:
                self.objects[elem_class] = elem
        
        self.promote_to_attributes(self.objects)

        # Sort according to custom order
        self.pipeline = Assembler.sort_pipeline(self.pipeline)
        
        self.pipeline_tracker = TrackIteration(self.pipeline, initTurn=0, finalTurn=-1, trackPeriods=self.tracking_periods)
        self.is_built = True
    

    def place_after(self, element: str, after: str) -> None:
        '''Place element right aftern another element in the track pipeline

        Args:
            element (_type_): _description_
            after (Union[str, Callable]): _description_
        '''

        if not self.is_built:
            print('Warning: Object not built. Call build method first.')
            return

        # Need to remove existing element from pipeline
        self.pipeline.remove(element)

        after_idx = self.pipeline.index(after)
        self.pipeline.insert(after_idx+1, element)

    def track(self, num_turns: int =1, with_timing: bool =False) -> None:
        '''Track all trackable pipeline objects for a number of turns

        Args:
            num_turns (int, optional): _description_. Defaults to 1.
            with_timing (bool, optional): _description_. Defaults to False.
        '''
        if not self.is_built:
            print('Warning: Object not built. Call build method first.')
            return

        if with_timing:
            # wrap all elements in the pipeline with timing
            pipeline = [timing.timeit(key=Assembler.get_function_name(stage))(stage) for stage in self.pipeline]
            self.pipeline_tracker._map = pipeline

        # Track all objects for the given number of turns
        self.pipeline_tracker(num_turns)

    def insert_at(self, element, index: int =0) -> None:
        '''Insert element at index, shifting all elements at and after the index to the right

        Args:
            element (_type_): _description_
            index (int, optional): _description_. Defaults to 0.
        '''
        self.element_list.insert(index, element)
        self.tracking_periods.insert(index, getattr(element, 'period', 1))
        self.is_built = False

    def append(self, element) -> None:
        '''Append element to the end of the element list

        Args:
            element (_type_): _description_
        '''
        self.insert(element, len(self.element_list), getattr(element, 'period', 1))

    def prepend(self, element) -> None:
        '''Prepend element to the beginning of the element list

        Args:
            element (_type_): _description_
        '''
        self.insert(element, 0, getattr(element, 'period', 1))

    def remove(self, element) -> None:
        '''Remove element from the element list

        Args:
            element (_type_): _description_
        '''
        index = self.element_list.index(element)
        del self.element_list[index]
        del self.tracking_periods[index]
        self.is_built = False

    def build_distribution(self, distribution) -> None:
        '''_summary_

        Args:
            distribution (_type_): _description_
        '''
        if Assembler.is_dictionary(distribution):
            assert len(distribution) == 1, 'Dictionary elements must be in the form: {disrtibution_type: {arg1: val1, arg2:val2, ...}}'
            distr_type, (distr_all_args) = next(iter(distribution.items()))
            assert distr_type in self.blond_distributions, f'Distribution type not recognized: {distr_type}'
            
            distr_args, distr_kwargs = Assembler.split_args_kwargs(distr_all_args)
            logger.debug(f'Found dictionary element with {len(distr_args)} positional arguments and {len(distr_kwargs)} kwargs')

            distr_args, distr_kwargs = self.replace_object_references(distr_args, distr_kwargs)
            self.blond_distributions[distr_type](*distr_args, **distr_kwargs)
        else:
            logger.debug(f'Distribution not recognized: {distribution}')
            
    
    def report_timing(self) -> None:
        '''Report timing information after tracking.
        '''
        timing.report()

    def to_yaml(self) -> None:
        '''Convert pipeline to yaml file.
        '''
        import yaml
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
