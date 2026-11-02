#!/bin/bash
# rerun qwen3.5 alone, not under parallel, capturing every kernel source this runner generates
M=${LLM_MODEL:-qwen3.5:0.8b}
REF=.github/debug_fp16/r_4_128_4_32_32.c
D=$(mktemp -d)
cat > "$D/cc.sh" <<'W'
#!/bin/bash
T=$(mktemp -p "$KERN_DIR" tmp.XXXXXX)
cat > "$T"
H=$(sha1sum "$T" | cut -c1-16)
mv -f "$T" "$KERN_DIR/$H.c"
clang "$@" < "$KERN_DIR/$H.c"
RC=$?
[ $RC -ne 0 ] && echo "$H rc=$RC" >> "$KERN_DIR/crashes.txt"
exit $RC
W
chmod +x "$D/cc.sh"
echo "== rerun alone"
echo "What's a male chicken called? Answer with only one word." | KERN_DIR=$D CC=$D/cc.sh CCACHE=0 timeout 240 python3 -m tinygrad.llm --no_chat_template --model "$M" 2>&1 | grep -E "rooster|Rooster|frontend command failed|Running pass 'Early" | sed -n 1,4p
echo "rerun exit ${PIPESTATUS[1]}, kernels $(ls "$D"/*.c 2>/dev/null | wc -l), crashes $(cat "$D/crashes.txt" 2>/dev/null | wc -l)"
cat "$D/crashes.txt" 2>/dev/null
echo "== kernel vs committed"
for K in $(grep -l 'r_4_128_4_32_32' "$D"/*.c 2>/dev/null); do
  cmp -s "$K" "$REF" && echo "$(basename "$K") identical to committed" || echo "$(basename "$K") DIFFERS from committed"
done
F="-c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident --target=x86_64-none-unknown-elf"
for i in 1 2 3; do clang $F -march=native - -o /dev/null < "$REF" >/dev/null 2>&1; echo "committed native try $i rc=$?"; done
exit 0
