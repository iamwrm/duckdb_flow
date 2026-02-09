cmake_minimum_required(VERSION 3.14)
project(duckdb_flow LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Shared RPATH so all executables can find libduckdb at build and install time
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")

# ── Prebuilt libduckdb ──────────────────────────────────────────────────────
include(FetchContent)

set(DUCKDB_VERSION "v1.4.0")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(DUCKDB_LIB_GLOB "libduckdb*.so")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(DUCKDB_ARCHIVE "libduckdb-linux-amd64.zip")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(DUCKDB_ARCHIVE "libduckdb-linux-arm64.zip")
    else()
        message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(DUCKDB_LIB_GLOB "libduckdb*.dylib")
    set(DUCKDB_ARCHIVE "libduckdb-osx-universal.zip")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Prefer a local archive when available; fall back to GitHub release
set(DUCKDB_LOCAL_ARCHIVE "${CMAKE_SOURCE_DIR}/${DUCKDB_ARCHIVE}")
if(EXISTS "${DUCKDB_LOCAL_ARCHIVE}")
    set(DUCKDB_URL "${DUCKDB_LOCAL_ARCHIVE}")
else()
    set(DUCKDB_URL
        "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/${DUCKDB_ARCHIVE}")
endif()

message(STATUS "Fetching DuckDB ${DUCKDB_VERSION} from ${DUCKDB_URL}")

FetchContent_Declare(
    duckdb_lib
    URL "${DUCKDB_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(duckdb_lib)

# Locate shared library (artifact names may include version suffixes)
file(GLOB DUCKDB_LIB_CANDIDATES "${duckdb_lib_SOURCE_DIR}/${DUCKDB_LIB_GLOB}")
if(NOT DUCKDB_LIB_CANDIDATES)
    message(FATAL_ERROR "Could not find DuckDB shared library in ${duckdb_lib_SOURCE_DIR}")
endif()
list(SORT DUCKDB_LIB_CANDIDATES)
list(GET DUCKDB_LIB_CANDIDATES 0 DUCKDB_LIB_PATH)

# Resolve duckdb.h -- download it when the prebuilt archive omits the header
if(EXISTS "${duckdb_lib_SOURCE_DIR}/duckdb.h")
    set(DUCKDB_INCLUDE_DIR "${duckdb_lib_SOURCE_DIR}")
else()
    set(DUCKDB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/duckdb_headers")
    file(MAKE_DIRECTORY "${DUCKDB_INCLUDE_DIR}")
    set(DUCKDB_HEADER_URL
        "https://raw.githubusercontent.com/duckdb/duckdb/${DUCKDB_VERSION}/src/include/duckdb.h")
    file(DOWNLOAD
        "${DUCKDB_HEADER_URL}"
        "${DUCKDB_INCLUDE_DIR}/duckdb.h"
        STATUS DUCKDB_HEADER_STATUS)
    list(GET DUCKDB_HEADER_STATUS 0 DUCKDB_HEADER_CODE)
    if(NOT DUCKDB_HEADER_CODE EQUAL 0)
        message(FATAL_ERROR
            "Failed to download duckdb.h from ${DUCKDB_HEADER_URL}: ${DUCKDB_HEADER_STATUS}")
    endif()
endif()

add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${DUCKDB_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INCLUDE_DIR}"
)

# Place the shared library next to our binaries so the RPATH resolves
file(COPY "${DUCKDB_LIB_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")

# ── Project targets ─────────────────────────────────────────────────────────

# Common compile/link settings shared by all executables
set(COMMON_LIBS duckdb_flow duckdb pthread)
set(COMMON_COMPILE_FLAGS -O2 -march=native)

# Header-only duckdb_flow library
add_library(duckdb_flow INTERFACE)
target_include_directories(duckdb_flow INTERFACE "${CMAKE_SOURCE_DIR}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${COMMON_LIBS})
target_compile_options(main PRIVATE ${COMMON_COMPILE_FLAGS})

add_executable(test_fuzz test.cpp)
target_link_libraries(test_fuzz PRIVATE ${COMMON_LIBS})
target_compile_options(test_fuzz PRIVATE ${COMMON_COMPILE_FLAGS})

# C11 compilation check — catches C++ syntax leaking back into the header
add_library(c11_check OBJECT c11_check.c)
target_link_libraries(c11_check PRIVATE duckdb_flow duckdb)
