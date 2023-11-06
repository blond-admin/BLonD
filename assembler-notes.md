## Notes
1. Problem with objects named as the classes: they hide the class name, therefore not possible to instantiate class
    FIXME
2. Allow for named or unnamed parameters
    DONE
    args can be in the form:
    * `'Beam' : {...}` (only kwargs)
    * or `'Beam' : (...)` (only args)
    * or `'Beam' : ((...), {...})` (both == tuple + 2 elems + first is tuple/iterable + second is dict)
    * Need a way to replace with references both named and unnamed parameters --> DONE

3. Process objects in order (re-arrange according to custom sort order?)
    TODO
4. Distributions can be passed as dictionary or as a tuple callable, (params)
    Params should be *args, **kwargs
    DONE
    
6. The assembler should export the initialized objects, so that you may access them as a class attribute
    DONE

7. Read from yaml:
    * How to convert arrays (could be simply array values)
    * Refernces to objects?
    * Objects in constructors?
8. Use the tracking_iteration class from Simon
    TODO


## Features
* Support for custom trackable objects
* Initialization with object list or Dictionary (yaml on-going)
* Automatically discover blond distributions (for dictionary initialization)
* Automatically discover blond classes (for dictionary initialization)
* Replace references to objects in dictionary initialization
* BLonD objects also availabel as Assembler attributes
* build_pipeline() to build the pipeline (ready to track)
* Convenient __str__ to print the assembler
* Tracking priority mechanism
* Supports insert/ append/ remove elements

* Uses TrackIteration (Simon) internally for tracking
* Integrated timing (with_timing=True)

## On-going/ Future Features
* Export to yaml
* Handle specific initialization options (i.e. for correct induced voltage tracking)
* Replace completely the operation of the tracker
    * Convert to a simple sequence of trackable objects
* Unittesting
* Examples




