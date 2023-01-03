# Sample Installation Guide: 

The installation guide has been prepared by Valay Agarawal for supercomputers in the University of Chicago and has been added here for support.

##Notes of pDMET installation: 

Last Edited: 20th Dec, 2021
This document is likely to change, please make sure that it is an updated version. 

## What does this document do? 
This document will help in installing pDMET in RCC3. 


Setting up the background, installation requirements. 
The following steps are required for installation. 
1. Get the yml file for setting up the environment
2. Get the bachrc file
3. Load required modules: python, intel, gcc, mkl using the commant
	``` 
	module load module-name
	```
4. Execute the bash file using 
	``` 
	source .bashrc
	```

## Installations
The following text is essentially what is discussed on the wannier90, pywannier and pdmet packages, but in an order that can be easy to understand. 

1. Prepare a directory in which you wish to install pdmet. 
	```
	mkdir Apps
	```

2. Install PySCF using the README file prepared by Matthew R. Hermes, Paul Calio, and Matthew Hennefarth. You only need to install PySCF, and other softwares are not required.
Note that the PySCF installation has a bit different yml and bashrc files, but the linked yml and bashrc should  be a superset of it, and hence should run without a problem.

3. Download wannier90 from  http://www.wannier.org/download/  (it is a gzipped.tar file)
   Place the download in the Apps folder. 
   You can use WinSCP for drag and drop type transfer. 
   Unzip the file using 
   ```
   tar -xf wannier90-xxx.tar.gz
   ```

   Note: In the file name ```wannier90-xxx.tar.gz```, ```xxx``` means the version of it. On 20th Dec, 2021 it is 3.1.0

4. We will now install pyWannier90, which acts as a phython wrapper for wannier90.
   ``` 
   git clone https://github.com/hungpham2017/pyWannier90.git       #Clone package from github
   cp pyWannier90/src/wannier_lib.F90 wannier90/src/wannier_lib.F90  # Replace
   ```
   Create ```make.inc``` file, which is linked here. 
   Complie wannier90-xxx
   ```
   make
   make lib
   ```

   If you get stuck in 
   ```make```, 
   please use the command 
   ```make clean``` 
   instead of 
   ```make```

   Change the path of libwannier90 in pyWannier/src/pywannier90.py file to '/home/user/Apps/pyWannier90/src'. Specifically, change the variable labelled as ```W90LIB```
   Test libwannier90 library:

   ```python -c "import libwannier90"		#Should return nothing if the compilation was successful```

5. Once completed, we will now proceed to installation of pDMET.
   ```
   git clone https://github.com/hungpham2017/pDMET.git    #Clone package from github, use the code fromt he dev branch (20th Dec 2021)
   cd pDMET/pdmet/lib
   rmdir -rf build
   mkdir build
   cd build
   ```
   Get the CMakeLists.txt file linked with the installation, and copy here. (You can again use WinSCP)
   
   ```
   export MKLROOT=/software/intel/parallel_studio_xe_2020_update1/mkl/ 
   export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
   cmake -DBLA_VENDOR=Intel10_64lp_seq .. 
   make 
   ```
6. You can also install mcu package for visualization of orbitals. 
   ```
   git clone https://github.com/hungpham2017/mcu.git
   ```
This should conclude the installation. To check:
   ```
   import python
   import pyscf
   import pdmet
   from pdmet import pdmet
   import pyWannier90
   ```

If these messages do not give any output, then your installation should be successful!