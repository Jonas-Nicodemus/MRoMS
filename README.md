# MRoMS

## About
This repository contains the exercises for the lecture 
[Model Reduction of Mechanical Systems](https://www.itm.uni-stuttgart.de/en/courses/lectures/modellreduktion-mechanischer-systeme/)
which was hold in the winter term 2020/21.

## Usage
Most of the exercises are jupyter notebooks.
Best practice to run a ```jupyter-notebook``` is:
* Install [```anaconda```](https://docs.anaconda.com/anaconda/install/)
* Create a conda environment , or use existing environment and activate it.
  ````shell script
  $ conda create --name <name (e.g. MRoMS)> python=3.8
  $ conda activate <name (e.g. MRoMS)>
  ````
* Install dependencies from ```requirements.txt```
  ````shell script
  $ pip install -r requirements.txt
  ````
* Install ```jupyter-notebook``` (if not already done) with
  ````shell script
  $ conda install jupyter
  ````
* Last part start the ```jupyter-notebook``` application with
  ````shell script
  $ jupyter-notebook
  ````
  and navigate to the ```*.ipynb``` file of your choice.

If you run in trouble with [```anaconda```](https://docs.anaconda.com/anaconda/install/),
the [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
might be useful.