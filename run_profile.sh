cd /Users/ernest.lee/Forges/see-through-cpp
rm -f profiling/spans.jsonl
./build/see-through -m models -i art/concept/anime_with_caption_cc0_0023.jpg -o /tmp/op_test.psd --res 256 --steps 1 2>&1 | grep -v -E "ggml_metal|UD |simdgroup|has |residency|shared|recommended|allocating|found|picking|deallocating|free:"