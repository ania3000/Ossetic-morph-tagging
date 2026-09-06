#!/bin/bash
# setup_environment.sh
git clone https://github.com/AlexeySorokin/Sigmorphon2018SharedTask.git
cd Sigmorphon2018SharedTask
python3 setup_aligner.py build_ext --inplace
cd pyparadigm
python3 setup.py build_ext --inplace
cd ../..
export PYTHONPATH=$PYTHONPATH:$(pwd)/Sigmorphon2018SharedTask
