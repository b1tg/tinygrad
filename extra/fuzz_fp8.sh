#!/bin/bash
for ((i=1; i<=10000; i++)); do
  # Place your commands here
  echo "Iteration number $i"
  pytest test/test_dtype.py::TestFp8sConversions::test_float_to_fp8s_fuzz -s
done