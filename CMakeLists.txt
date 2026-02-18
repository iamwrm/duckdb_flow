cmake_minimum_required(VERSION 3.14)
project(duckdb_flow LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}")

# ── Fetch prebuilt libduckdb ────────────────────────────────────────────────
include(FetchContent)

set(DUCKDB_VERSION "v1.4.4")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(DUCKDB_ARCHIVE "libduckdb-linux-amd64.zip")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(DUCKDB_ARCHIVE "libduckdb-linux-arm64.zip")
else()
    message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/${DUCKDB_ARCHIVE}")
    set(DUCKDB_URL "${CMAKE_SOURCE_DIR}/${DUCKDB_ARCHIVE}")
else()
    set(DUCKDB_URL "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/${DUCKDB_ARCHIVE}")
endif()

FetchContent_Declare(duckdb_lib URL "${DUCKDB_URL}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(duckdb_lib)

add_library(duckdb SHARED IMPORTED GLOBAL)
set_target_properties(duckdb PROPERTIES
    IMPORTED_LOCATION "${duckdb_lib_SOURCE_DIR}/libduckdb.so"
    INTERFACE_INCLUDE_DIRECTORIES "${duckdb_lib_SOURCE_DIR}"
)
file(COPY "${duckdb_lib_SOURCE_DIR}/libduckdb.so" DESTINATION "${CMAKE_BINARY_DIR}")

# ── Targets ─────────────────────────────────────────────────────────────────

# Header-only interface library so targets can #include "duckdb_flow.h"
add_library(duckdb_flow INTERFACE)
target_include_directories(duckdb_flow INTERFACE "${CMAKE_SOURCE_DIR}")

add_executable(main main.cpp)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(main PRIVATE duckdb_flow duckdb pthread)
target_compile_options(main PRIVATE -O2 -march=native)

add_executable(test_fuzz tests/test.cpp)
target_include_directories(test_fuzz PRIVATE "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(test_fuzz PRIVATE duckdb_flow duckdb pthread)
target_compile_options(test_fuzz PRIVATE -O2 -march=native)

# Ensures duckdb_flow.h stays valid C11 (catches C++ syntax creep)
add_library(c11_check OBJECT c11_check.c)
target_link_libraries(c11_check PRIVATE duckdb_flow duckdb)

add_executable(udf_bench udf/main.cpp)
target_link_libraries(udf_bench PRIVATE duckdb)
target_compile_options(udf_bench PRIVATE -O2 -march=native -Wall -Wextra)
