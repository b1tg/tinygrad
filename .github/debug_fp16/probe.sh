#!/bin/bash
K=.github/debug_fp16/r_4_128_4_32_32.c
F="-c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident --target=x86_64-none-unknown-elf"
echo "== cpu"
grep -m1 "model name" /proc/cpuinfo
echo "nproc $(nproc)"
free -g | sed -n 2p
echo "avx512_fp16 in /proc/cpuinfo: $(grep -m1 -c avx512_fp16 /proc/cpuinfo)"
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx512|amx)' | tr '\n' ' '; echo
echo "== clang"
which clang
clang --version | head -1
clang -march=native -### -x c -c /dev/null 2>&1 | tr ' ' '\n' | grep -A1 '"-target-cpu"' | tr '\n' ' '; echo
clang -march=native -### -x c -c /dev/null 2>&1 | tr ' ' '\n' | grep -o '.avx512fp16' | sed 's/^/resolved feature: /'
echo "== crash kernel"
run() { clang $F "$@" - -o /dev/null < $K >/dev/null 2>&1; echo "rc=$? : $*"; }
run -march=native
run -march=native -mno-avx512fp16
run -march=sapphirerapids
run -march=sapphirerapids -mno-avx512fp16
