## Folder contents
- [simulation_args.txt](simulation_args.txt)
    - A list of inputs for which to execute the simulation
- [simulation_run.sh](simulation_run.sh)
    - Glue code to connect a python file with HTCondor arguments and point to virtual environment
    - Executing this file alone will not take advantage of HTCondor.
- [config_condor.sub](config_condor.sub)
  - Configuration of what to send to HTCondor
  - Number of arguments in `simulation_args.txt` must match to the last line of this file 
- [submit_condor.sh](submit_condor.sh)
    - Shortcut to execute the simulation on HTCondor
- [logs](logs)
  - Logs of HTCondor execution are stored here (defined in `submit_condor.sh`)

More elaborate scripts can be found at [BLonD Submission Scripts](https://gitlab.cern.ch/blond/submission-scripts).