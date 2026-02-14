cmake_minimum_required(VERSION 3.14)
project(duckdb_udf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RPATH so the executable finds libduckdb next to it
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")

# ── Fetch prebuilt libduckdb ────────────────────────────────────────────────
include(FetchContent)

set(DUCKDB_VERSION "v1.4.0")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DUCKDB_LIB_GLOB "libduckdb*.so")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(DUCKDB_ARCHIVE "libduckdb-linux-amd64.zip")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(DUCKDB_ARCHIVE "libduckdb-linux-arm64.zip")
    else()
        message(FATAL_ERROR "Unsupported Linux arch: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(DUCKDB_LIB_GLOB "libduckdb*.dylib")
    set(DUCKDB_ARCHIVE "libduckdb-osx-universal.zip")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(DUCKDB_URL
    "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/${DUCKDB_ARCHIVE}")
message(STATUS "Fetching DuckDB ${DUCKDB_VERSION} from ${DUCKDB_URL}")

FetchContent_Declare(
    duckdb_lib
    URL "${DUCKDB_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(duckdb_lib)

# Find shared library
file(GLOB DUCKDB_LIB_CANDIDATES "${duckdb_lib_SOURCE_DIR}/${DUCKDB_LIB_GLOB}")
if(NOT DUCKDB_LIB_CANDIDATES)
    message(FATAL_ERROR "DuckDB shared library not found in ${duckdb_lib_SOURCE_DIR}")
endif()
list(SORT DUCKDB_LIB_CANDIDATES)
list(GET DUCKDB_LIB_CANDIDATES 0 DUCKDB_LIB_PATH)

# Resolve header
if(EXISTS "${duckdb_lib_SOURCE_DIR}/duckdb.h")
    set(DUCKDB_INCLUDE_DIR "${duckdb_lib_SOURCE_DIR}")
else()
    set(DUCKDB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/duckdb_headers")
    file(MAKE_DIRECTORY "${DUCKDB_INCLUDE_DIR}")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/duckdb/duckdb/${DUCKDB_VERSION}/src/include/duckdb.h"
        "${DUCKDB_INCLUDE_DIR}/duckdb.h"
        STATUS _st)
    list(GET _st 0 _code)
    if(NOT _code EQUAL 0)
        message(FATAL_ERROR "Failed to download duckdb.h: ${_st}")
    endif()
endif()

# Import as shared library
add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${DUCKDB_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INCLUDE_DIR}"
)
file(COPY "${DUCKDB_LIB_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")

# ── Build ───────────────────────────────────────────────────────────────────
add_executable(main main.cpp scalar.cpp simd.cpp)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(main PRIVATE duckdb)

# Common flags
target_compile_options(main PRIVATE -O2 -march=native -Wall -Wextra)

# scalar.cpp: prevent all auto-vectorization so it stays truly scalar
set_source_files_properties(scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize"
)
