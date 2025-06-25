# Sea/Air/Cargo Container Pathway Simulation System

## Introduction
The package simulates the current and future (either with SCHS or CSP1/CSP3
sampling protocols for the CAL containers) states of external shipping container
contamination risk management. The flowcharts of both states are shown below:

![Image](https://github.com/user-attachments/assets/62912066-fa28-41d3-9cd2-175b6ae1ad00)

![Image](https://github.com/user-attachments/assets/fee218ea-b88f-4441-997c-70bb96f5ad9d)


## Repo structure
The tests folder contains unit tests.
The flatpack package consists of the preproc,  src and plot modules
inside the flatpack folder. The preproc module comprises two
submodules for preprocessing, i.e., generating the virtual containers
consistent with the conditional probability table (CPT), which should
have four container attribute (CALSourceCountry status, SCHS, RuralDest
and TypeOfGoods) as columns, and prop and p\_yes columns for the proportion
and contamination probability, respectively. Before the containers can
be created CALSourceCountryMedium attribute has to be added.

The src module is the "engine" of the package, which holds the implementation
of the current and future container pathway.

The plot module provides the utilities which generate plots from the pathway
simulation results as bar-charts and pie-charts.

The pipeline module contains a single file which implements the full workflow
in a couple of lines of code.

The data folder has single CSV file with the parameters to carry out a random
sampling for the cargo compliance verification (CCV) of low-risk containers.

The notebooks folder has several notebooks demonstrating the usage of this package.

The Flowcharts folder has illustrations of the current and future pathway.

The samples folder provides examples of YAML input files and CPTs as CSV files.

## Installation
The install, clone the package from the github repo
https://github.com/snazzybloke/fatpack
Check out the main branch in the repo folder.

This package has been successfully tested using Python 3.12.
It is advisable (but not necessary) to install it in a dedicated virtual environment.
The latter can be created using the Python built-in module venv:
> python -m venv path\_to\_new\_virtual\_environment

If using Anaconda it can be created with:
conda create -n a\_new\_environmet.

Then activate the newly created virtual environment.
It is adviseable to upgrade the pip module (the Python package manager):
> python -m pip install -U pip

In the same environment install the pip-tools package using pip:
> python -m pip install pip-tools

Then, from the repo folder, execute:
> python -m piptools compile

This will create a new file requirements.txt, which is based on the file requirements.in
provided with the repo.
The new file serves to satisfy the package dependencies inside your virtual environment.

Finally, we can install flatpack by executing the following command, again from the repo:
> python -m pip install .

## Unit testing
Unit testing using is carried out using the pytest package, which can be installed with:
> python -m pip install pytest pytest-cov coverage

In Azure DataBricks the following bash variables should be set to execute the tests (optional elsewhere):
> export PYTHONPYCACHEPREFIX=/tmp/pycache

> export TMPDIR=/tmp

Then execute the following command (also inside the repo folder) in a terminal app:
> python -m pytest ./tests

or for a verbose output:

> python -m pytest ./tests -v -s

For coverage report, using the pacakges pytest-cov and coverager, and execute:
> python -m pytest --cov ./tests

Due to the use of random number generators one or even two tests may occasionally yield an error.
If this happens, simply repeat the command and the tests should be 100% successful.

## Using the package flatpack
All the modules of the package require an input configuration with necessary parameters.
The folder flatpack22/samples provides several YAML files.

To run a simple workflow in a python shell or jupyter notebook start by loading the
Confing class from the flatpack.src.config module:
> from flatpack.src.config import Config

> conf = Config("InputParameters.yaml")

Now that the Config-type object (named "conf") has been defined, one can perform pre-processing
if required. To this end two files are provided in the flatpack.preproc module.
As the first step, the medium\_risk\_ext.py is used to add the "CALSourceCountryMedium"
binary attribute (for medium risk when True, as opposed to high risk otherwise)
to the top conditional probability table. An example of the top CPT is the file
cpt\_top.csv in the folder flatpack22/samples, where we can also find the result of its
preprocessing in the file cpt\_top\_with\_med\_KEEP.csv.
> from flatpack.preproc import medium\_risk\_ext

> medium\_risk\_ext.main(conf)

Once we have generated the CPT with the "CALSourceCountryMedium" attribute, we can generate
the virtual containers using the make\_containers.py sub-module.
> from flatpack.preproc import make\_containers

> make\_containers.main(conf)

Once we have the CSV file with the desired number of virtual containers and their attributes
we can continue with the actual processing. This is done using the sub-modules
simulate\_pathways and multi\_simulate\_pathways inside the flatpack.src module.
To execute a single simulation of the current and future container pathway:
> from flatpack.src import simulate\_pathways

> simulate\_pathways.main(conf)

To generate the plots of the results, use the flatpack.plot module:
> from flatpack.plot import plot\_data

> plot\_data.run\_individual\_outputs(conf)

To run a multi-simulation workflow:
> from flatpack.src import multi\_simulate\_pathways

> multi\_simulate\_pathways.main(conf)

To generate the plots of the results, use the plot\_multisim\_data sub-module:
> from flatpack.plot import plot\_multisim\_data

> plot\_multisim\_data.main(conf)

Alternatively, the whole workflow can be executed from a terminal app simply with:
> python path\_to\_/flatpack/pipeline/full\_wflow.py -y InputParameters.yaml

(Note: InputParameters.yaml is just a dummy name, replace it with the actual file name).
Finally, a browser-based GUI is provided for the ease of use via the flatpack22b/pipeline/gui\_form.py script.
In order to run it one needs to install the streamlit package first with
> python -m pip streamlit

Then the GUI starts with
> python -m streamlit run path\_to\_/flatpack22b/pipeline/gui\_form.py


## Package removal
To remove the installed package simply execute:
> python -m pip uninstall flatpack
