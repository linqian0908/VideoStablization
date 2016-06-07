TEMPLATE = app
TARGET   = stablization_test

MODULE_DEPS = stablization

LIB_DEPS   += opencv

include(common.pri)

SOURCES += \
    videostablization/stablization_test.cpp \


