
# LXPLUS GUIDE

## .bashrc modification
Add to your bashrc the following lines
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
## Creating sub files
To submit a job to lxplus you need a sub file and an executable.

## Running an example
To run a main file like minimalworking.py, you create a sub file like `run_example.sub` and you submit it with:  
```
condor_submit run_example.sub
```
while having `run_example.sh` in your directory. Modify `run_example.sh` before you try it.



