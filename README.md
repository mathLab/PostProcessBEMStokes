

# PostProcessBEMStokes: parallel flow field post processor for BEMStokes 


The library represents a parallel post processor for the Stokes system solver [BEMStokes](https://github.com/mathLab/BEMStokes). We have developed the software in C++ on top of many high performance libraries, the [deal.II](https://github.com/dealii/dealii) library for Finite Element Handling, the [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download) project and [Trilinos](https://trilinos.org/) library for automatic Workload balance, and [deal2lkit](https://github.com/luca-heltai/deal2lkit) for parameter handling. 

## Provided features

We provide the following capabilities

- reconstruction of the flow field on a user defined series of points (in a text file)
- reconstruction of the flow field on up to four user defined surfaces defined in the parameter file
- reconstruction of the flow field on a user defined box, in this case it is also possible to compute the dissipation energy scalar field
- computation of the force dipoles up to arbitrary order to define the swimmer behavior (pusher puller)

## Code Structure
We have subdivided the code in main classes to handle the many different aspects of a complete BEM simulation.

- PostProcessBEMStokes. This class is in charge of organising the overall post process flow field reconstruction. It has interfaces with all the other classes in order to perform a complete simulation.
- FreeSurfaceStokesKernel. A stokes kernel for free surface interfaces
- StokesKernel. container for the Stokes fundamental solutions.


## Install Procedure from scratch
In order to successfully compile the code you need 

- to install [BEMStokes](https://github.com/mathLab/BEMStokes) with all its dependencies

For greater details you can look to the installation procedure of BEMStokes.


### PostProcessBEMStokes Installation procedure

Then you can clone the repository and compile it

	git clone https://github.com/nicola-giuliani/PostProcessBEMStokes.git
	cd BEMStokes
	mkdir build
	cd build
	cmake ../
	make -j4

After you have compiled your application, you can run 

	make test

or
	
	ctest 

to start the testsuite.

Take a look at
https://www.dealii.org/developer/developers/testsuite.html for more
information on how to create tests and add categories of tests.

If you want you can run a preliminary execution in the build library typing
	
	mpirun -np 1 pp_bem_stokes_2d
	
this will automatically generate the parameter file for the bi-dimensional run while 

	mpirun -np 1 pp_bem_stokes_3d
	
will create a proper parameter file for a 3 dimensional simulation.

#Notice to developers

Before making a pull request, please make sure you run the script

    ./scripts/indent

from the base directory of this project, to ensure that no random 
white space changes are inserted in the repository.

The script requires Artistic Style Version 2.04 (astyle) to work 
properly.

#Licence

Please see the file [LICENSE]() for details



