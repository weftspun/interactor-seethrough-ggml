cd /Users/ernest.lee/Forges/see-through-cpp
rm -f profiling/spans.jsonl
./build/see-through -m models -i art/concept/anime_with_caption_cc0_0023.jpg -o /tmp/op_test.psd --res 256 --steps 1 --depth-steps 1 2>&1 | grep -vE "ggml_metal|loaded kernel|compiling pipeline|GPU family|simdgroup|has |residency|shared buffers|recommended|allocating|found device|picking|use |deallocating|free:"
echo ""
python3 -c "
import json
with open('profiling/spans.jsonl') as f:
    for l in f:
        if l.strip():
            s=json.loads(l)
            d=s.get('duration_ms',0)
            print(f'  {d:10.1f} ms  {s[\"name\"]}')
" 2>/dev/null
echo "EXIT: done"
