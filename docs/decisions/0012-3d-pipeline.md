1. Text generate or fetch 2d concept art.
2. Apply https://replicate.com/qwen/qwen-image-edit-plus
  * Insert `docs\decisions\attachments\0904bf5b-e904-4a71-a116-b4a541daa1f3.png` into image index 0
  * Insert step one's image into image into 1.
  * Add prompt ```The character in image 2 adopts the colours from image 1 facing forward.```
3. Run see-through-cpp
4. Post process psd
  * Remove eyes, eyebrows, irides, mouth, nose, skin, back hair
  * Keep neck, front hair and face
5. Apply https://huggingface.co/spaces/TencentARC/Pixal3D
  * Use maximum decimation
6. Apply https://huggingface.co/spaces/VAST-AI/SkinTokens
7. Use https://mesh2motion.org/ or Nvidia Kimodo.
