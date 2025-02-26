# this file is intended to be executed inside the docker container
# to test the docker container before shipping

git clone https://gitlab.cern.ch/blond/BLonD/
cd BLonD
# compile blond
python3 blond/compile.py -p --optimize --parallel
pip install .
# Run all unittests + the examples
python3 -m pytest --exitfirst -v unittests/ --ignore=unittests/gpu/ --ignore=unittests/integration/test_gpu_examples --ignore=unittests/integration/test_validate_gpu