# StanShock
Author: Kevin Grogan

Contained in this folder is StanShock v0.1. StanShock is a quasi-1D gas dynamics solver designed model shock tube experiments. 

The provided version stanShock has the following capabilities:

	Variable cross-sectional area
	Boundary layer modeling
	Multicomponent gas interfaces
	
	Reaction Chemistry
	Species and thermal diffusion
	Geometric Optimization
	

StanShock is writen in object-oriented python, which allows the client to flexibly script and run stanShock cases. StanShock leverages several python modules listed below (tested version number given):

	python 3.7.1
	numpy 1.15.4 (common python package for scientific computations)
	numba 0.41.0 (just-in-time compilation for significant speed-up)
	cantera 2.4.0 (encapsulates the thermodynamics and kinetics)
	matplotlib 3.0.2 (plotting module)
	sciPy 1.1.0 (module with additional common numerical algorithms)

It is recommended to install an anaconda distribution (https://www.continuum.io/downloads), which will contain all dependencies except cantera. Cantera (http://www.cantera.org/docs/sphinx/html/index.html) will require a separate installation.

Included are six examples:

	laminarFlame (laminar flame test case of stoichiometric H2/Air)
	optimization (driver insert optimization)
	validationCases (four validation test cases)
		case1 (baseline)
		case2 (step change in driver/driven area)
		case3 (driver insert case)
		case4 (disparate driver/driven mixtures)

Files include:

	stanShock.py (entirety of the StanShock solver code)
	*.{xml,cti} (cantera files containing the thermodiffusive properties)
	{laminarFlame,optimization,case{1..4}}.py (python driver scripts)
	case{1..4}.csv (experimental shock tube data for the validation cases)
	*.pyc (compiled python code)
  
Please report any issues or bugs to the author at kevin.p.grogan@gmail.com. 
