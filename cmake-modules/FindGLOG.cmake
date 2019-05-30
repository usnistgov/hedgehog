# - Try to find Glog
#
# The following variables are optionally searched for defaults
#  GLOG_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  GLOG_FOUND
#  GLOG_INCLUDE_DIRS
#  GLOG_LIBRARIES
#

include(FindPackageHandleStandardArgs)

set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")

if (WIN32)
    find_path(GLOG_INCLUDE_DIR glog/logging.h
            PATHS ${GLOG_ROOT_DIR}/src/windows)
else ()
    find_path(GLOG_INCLUDE_DIR glog/logging.h
            PATHS ${GLOG_ROOT_DIR})
endif ()

find_library(GLOG_LIBRARY glog
        PATHS ${GLOG_ROOT_DIR}
        PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(GLOG DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

if (GLOG_FOUND)
    set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
    set(GLOG_LIBRARIES ${GLOG_LIBRARY})
    message(STATUS "Found GLOG    (include: ${GLOG_INCLUDE_DIR}, library: ${GLOG_LIBRARY})")
endif ()

mark_as_advanced(GLOG_ROOT_DIR GLOG_LIBRARY GLOG_INCLUDE_DIR)