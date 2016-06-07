#############################################################
## Main configuration
#############################################################

#CMAKE_BIN = cmake

#LIB_BUILD_DIR = build-linux

## Intermedianate results
##OBJECTS_DIR = $${ROOT_BUILD_DIR}/
##MOC_DIR     = $${ROOT_BUILD_DIR}/
##RCC_DIR     = $${ROOT_BUILD_DIR}/
##UI_DIR      = $${ROOT_BUILD_DIR}/


#############################################################
## Complier and linker configuration
#############################################################
#CONFIG          -= qt
#QMAKE_CXXFLAGS  += -std=c++11
#QMAKE_CXXFLAGS_WARN_ON -= -Wall
#QMAKE_CXXFLAGS         += -Wall -Wno-strict-overflow


#############################################################
## Add gmock/gtest header and library dependencies
#############################################################

#RUN_UNIT_TEST {
#    INCLUDEPATH += /usr/include/gmock
#    INCLUDEPATH += /usr/include/gtest
#    LIBS += -lgmock_main
#    LIBS += -lgmock
#}



#############################################################
## Add library dependencies
#############################################################

#for(lib, MODULE_DEPS){
#    LIBS += -l$${lib}

#    PRE_TARGETDEPS += lib$${lib}.so

#    INCLUDEPATH += $${lib}
#    DEPENDPATH  += $${lib}
#}



#############################################################
## Add library dependencies
#############################################################

##opencv
#contains(LIB_DEPS,opencv){
#    OPENCV_LIBS = $$system(pkg-config opencv --libs)
#    LIBS += $$OPENCVLIBS
#}


################################################################################
# Main configuration / path dependencies
################################################################################

# CMake binary link
CMAKE_BIN = cmake

# OS prefix for library build directories
macx {
    OS_NAME = osx
} else:win32 {
    OS_NAME = win
} else {
    OS_NAME = linux
}

LIB_BUILD_DIR = build-$${OS_NAME}

#CONFIG(release, debug|release) {
#    DESTDIR = $${ROOT_SRC_DIR}/../build-release
#}

#CONFIG(debug, debug|release) {
#    DESTDIR = $${ROOT_SRC_DIR}/../build-debug
#}

# Put intermediates into a subdir
OBJECTS_DIR = $${ROOT_BUILD_DIR}/out
MOC_DIR     = $${ROOT_BUILD_DIR}/out
RCC_DIR     = $${ROOT_BUILD_DIR}/out
UI_DIR      = $${ROOT_BUILD_DIR}/out

################################################################################
# Compiler and linker configuration
################################################################################

CONFIG                     += thread
CONFIG                     += object_parallel_to_source
CONFIG                     -= qt
DEFINES                    += ARCH_X86 _FILE_OFFSET_BITS=64
QMAKE_CXXFLAGS             += -std=c++11
macx {
    INCLUDEPATH            += /usr/local/include /opt/local/include
    LIBS                   += -L/usr/local/lib -L/opt/local/lib -lc++
    QMAKE_CXXFLAGS_WARN_ON -= -Wall
    QMAKE_CXXFLAGS         += -stdlib=libc++ -Wall -Wno-unknown-pragmas
    DEFINES                += OS_APPLE_OSX
    QMAKE_MAC_SDK           = macosx10.11

    QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.7
} else {
    QMAKE_CXXFLAGS_WARN_ON -= -Wall
    QMAKE_CXXFLAGS         += -Wall -Wno-strict-overflow
    DEFINES                += OS_LINUX
}

# max performance without unsafe floating-point optimizations
RELEASE_FLAGS           = -ffast-math -fno-unsafe-math-optimizations
contains(CODE_PROFILING,yes) {
    RELEASE_FLAGS      += -O2
    SHARED_C_FLAGS     += -g -fno-omit-frame-pointer
    DEFINES            += ENABLE_CODE_PROFILING
} else {
    RELEASE_FLAGS      += -O3
}


SHARED_C_FLAGS          += -march=core-avx2 -fPIC -ffunction-sections -fdata-sections
SHARED_CXX_FLAGS        += $${SHARED_C_FLAGS} -fvisibility-inlines-hidden
macx {
    SHARED_LFLAGS           += -Wl,-dead_strip
} else {
    SHARED_LFLAGS           += -Wl,--gc-sections
}

QMAKE_CXXFLAGS_RELEASE   = $${RELEASE_FLAGS} $${SHARED_CXX_FLAGS}
QMAKE_CXXFLAGS_DEBUG    += $${SHARED_CXX_FLAGS}
QMAKE_CFLAGS_RELEASE     = $${RELEASE_FLAGS} $${SHARED_C_FLAGS}
QMAKE_CFLAGS_DEBUG      += $${SHARED_C_FLAGS}
macx {
    QMAKE_LFLAGS_RELEASE = $${RELEASE_FLAGS} $${SHARED_CXX_FLAGS} $${SHARED_LFLAGS}
} else {
    QMAKE_LFLAGS_RELEASE = -s $${RELEASE_FLAGS} $${SHARED_CXX_FLAGS} $${SHARED_LFLAGS}
}
LIBS                    += -L$${ROOT_BUILD_DIR} -ldl


################################################################################
# Add gmock/gtest header and library dependencies
################################################################################

RUN_UNIT_TESTS {
    # OSX specific configuration
    macx {
        #INCLUDEPATH += /usr/local/include
        INCLUDEPATH += /usr/include
        #LIBS += -L/usr/local/lib
        LIBS += -L/usr/lib
        LIBS += -lc++
    } else {
        INCLUDEPATH += /usr/share/gmock-1.7.0/include
        INCLUDEPATH += /usr/share/gmock-1.7.0/gtest/include
        LIBS += -L${GMOCK_HOME}/mybuild -L${GMOCK_HOME}/gtest/mybuild
    }
    LIBS += -lgmock_main -lgtest
}

################################################################################
# Module dependencies
################################################################################

for(lib, MODULE_DEPS) {
    # link with this
    LIBS           += -l$${lib}
    # make the target dependent on the library
    macx {
        PRE_TARGETDEPS += lib$${lib}.1.dylib
    } else {
        PRE_TARGETDEPS += lib$${lib}.so
    }
    # and search its directory for headers
    INCLUDEPATH += $${lib}
    DEPENDPATH  += $${lib}
}

################################################################################
# Library definitions
################################################################################

# opencv
contains(LIB_DEPS, opencv) {
    OPENCV_LIBS = $$system(pkg-config opencv --libs)
    LIBS += $$OPENCV_LIBS
}

LIBS += -L"/opt/lp_solve_5.5/lpsolve55/bin/ux64/" -llpsolve55
INCLUDEPATH += /opt/lp_solve_5.5
DEPENDPATH  += /opt/lp_solve_5.5
