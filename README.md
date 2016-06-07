## Repository contents for L1 Path Smoothing for Video Optimization

| file | description |
| --- | --- |
| 'videostab.cpp[h]' | the main function for running L1 path smoothing |
| 'stablization_test.cpp' | runs videostab |
| 'videostablization.pro'| Qt project files |
| 'common.pri', 'libstablization.pro','stablization_test.pro' | more Qt project files |

Dependency: lp-solve, opencv3.0, cvplot

Using lp-solve:

1. download from http://lpsolve.sourceforge.net/5.5/
 
2. unzip and cd to the folder (suppose it is lp_solve_5.5/ )

3. cd lpsolve55
    sh ccc  (and you should see folder bin/ux64/)

4. include the library in the project
    in qt, change the following lines in libstablization.pro to actual lp_solve_5.5 path

    LIBS += -L"/home/memgrapher/Code/lp_solve_5.5/lpsolve55/bin/ux64/" -llpsolve55
    INCLUDEPATH += /home/memgrapher/Code/lp_solve_5.5
    DEPENDPATH  += /home/memgrapher/Code/lp_solve_5.5
    
## Repository contents for L2 video smoothing,face saliency constraint and face decoration

| file | description |
| --- | --- |
| `VideoStablization.cpp[h,vcxproj]` | the main functions |
| `CMakeLists.txt` | cmake |
| `cvplot`  | lib files for cvplot, used for plotting trajectory |

Dependency: OpenFace, opencv3.0, cvplot

download OpenFace from https://github.com/TadasBaltrusaitis/OpenFace and follow instruction.

## Data files

| file | description |
| --- | --- |
| `SANY0025.avi`, 'gleicher1.m4' | example shaky video |
| `L1-X.png`, 'L1-Y.png', 'L1-Angle.png' | L1-optimized path from running videostab on SANY0025.avi. Note to TA and Instructor: the figure in our report submitted on Gradescope is incorrect. These are the correct version |

## L1 usage

put videostab.cpp[h] and stablization_test.cpp in videostab/, and import videostab.pro into Qt. Change input and output file path in stablization_test.cpp.

## VideoStablization usage

in build folder, run make, and then

    ./bin/VideoStablization -f [input file] [-salient 0.9] [-crop 0.1] [-pathsmooth 20] [-facesmooth 5]
    
default value: 

sallient = 0 //not use face feature

hcrop =0.1 //crop 10% of wide and height on both edge

pathsmooth=20 //radius for smoothing trajectory.

facesmooth=5 //radius for smoothing face feature to center

