# The interactor contract.
#
# `weft::interactor` links nothing, and that is the test. If a socket library or a QUIC
# implementation ever appears below, the boundary this repository exists to hold has gone, and
# the linker is a better place to find that out than a review is.
cmake_minimum_required(VERSION 3.22)
project(weft_interactor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(weft_interactor STATIC src/cbor.c)
add_library(weft::interactor ALIAS weft_interactor)
target_include_directories(weft_interactor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(weft_interactor PRIVATE -Wall -Wextra)

# Vendored into a service as a subtree, this file is included rather than run, so the proof
# only builds when this repository is the top of the tree. A service's build is not the place
# to compile somebody else's test.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  enable_testing()
  add_executable(weft_interactor_roundtrip proof/roundtrip.c)
  target_link_libraries(weft_interactor_roundtrip PRIVATE weft_interactor)
  target_compile_options(weft_interactor_roundtrip PRIVATE -Wall -Wextra)
  add_test(NAME roundtrip COMMAND weft_interactor_roundtrip)
endif()
