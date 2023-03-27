cruft update -c container
autopep8 -v --in-place -r .
git status


mysub.py -limit 1:00 -m 3 -j condaCreate -- \
   conda env create -p ~/scratch/conda/$$ --file requirements_dev.yaml
