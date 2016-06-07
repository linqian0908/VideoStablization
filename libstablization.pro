TEMPLATE = lib
TARGET   = stablization

LIB_DEPS   += opencv

include(common.pri)

HEADERS += \
    videostablization/videostab.h

SOURCES += \
    videostablization/videostab.cpp

