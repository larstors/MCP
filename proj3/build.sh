#!/bin/sh

# Basic compiler arguments; we use clang++ by default but if you want
# to use eg g++ then set the CXX environment variable
compile="${CXX:-clang++} -Wall -Werror -std=c++17"

debug=0
# Iterate over script arguments
for arg in "$@"; do
  case $arg in
    --debug)
    debug=1
    ;;
  esac
done

if [ $debug -ne 0 ]; then
  echo "Building with debugging enabled"
  $compile -o MD MD.cpp
else
  echo "Building with optimisations enabled; use the --debug option if you suspect problems"
  $compile -O0 -g -DNDEBUG -o MD MD.cpp
fi