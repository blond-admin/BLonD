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





