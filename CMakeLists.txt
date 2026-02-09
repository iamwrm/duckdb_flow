cmake_minimum_required(VERSION 3.14)
project(duckdb_doublebuf LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

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
        message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(DUCKDB_LIB_GLOB "libduckdb*.dylib")
    # DuckDB ships a universal macOS library artifact.
    set(DUCKDB_ARCHIVE "libduckdb-osx-universal.zip")
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

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

# Locate shared library (artifact names can be versioned)
file(GLOB DUCKDB_LIB_CANDIDATES
     "${duckdb_lib_SOURCE_DIR}/${DUCKDB_LIB_GLOB}")
list(LENGTH DUCKDB_LIB_CANDIDATES DUCKDB_LIB_COUNT)
if(DUCKDB_LIB_COUNT EQUAL 0)
    message(FATAL_ERROR "Could not find DuckDB shared library in ${duckdb_lib_SOURCE_DIR}")
endif()
list(SORT DUCKDB_LIB_CANDIDATES)
list(GET DUCKDB_LIB_CANDIDATES 0 DUCKDB_LIB_PATH)

# Resolve duckdb.h (not always shipped in the prebuilt shared-lib archive)
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

# Create imported target for DuckDB
add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${DUCKDB_LIB_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${DUCKDB_INCLUDE_DIR}"
)

# Install the shared library alongside our binaries so rpath works
file(COPY "${DUCKDB_LIB_PATH}"
     DESTINATION "${CMAKE_BINARY_DIR}")

# ── Our targets ─────────────────────────────────────────────────────────────

# Header-only doublebuf library
add_library(doublebuf INTERFACE)
target_include_directories(doublebuf INTERFACE "${CMAKE_SOURCE_DIR}")

# Main demo
add_executable(main main.cpp)
target_link_libraries(main PRIVATE doublebuf duckdb pthread)
target_compile_options(main PRIVATE -O2 -march=native)
set_target_properties(main PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}"
)

# Fuzz test suite
add_executable(test_fuzz test.cpp)
target_link_libraries(test_fuzz PRIVATE doublebuf duckdb pthread)
target_compile_options(test_fuzz PRIVATE -O2 -march=native)
set_target_properties(test_fuzz PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
    INSTALL_RPATH "${CMAKE_BINARY_DIR}"
)
