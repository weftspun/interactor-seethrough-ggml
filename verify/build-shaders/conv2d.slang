[[vk::binding(0, 0)]]
StructuredBuffer<float> input;
[[vk::binding(1, 0)]]
StructuredBuffer<float> weight;
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> output;
[[vk::binding(3, 0)]]
uint OC;
[[vk::binding(4, 0)]]
uint OH;
[[vk::binding(5, 0)]]
uint OW;
[[vk::binding(6, 0)]]
uint C;

[shader("compute")] [numthreads(8, 8, 1)]
void conv2d(uint3 dtid : SV_DispatchThreadID) {
  uint oc = dtid[0u];
  uint oh = dtid[1u];
  uint ow = dtid[2u];
  float sum = 0.000000;
  for (uint c = 0u; c < C; ++c) {
    for (uint kh = 0u; kh < 3u; ++kh) {
      for (uint kw = 0u; kw < 3u; ++kw) {
        sum = (sum + (input[((c * (OH * OW)) + ((((oh * 1u) + kh) * OW) + ((ow * 1u) + kw)))] * weight[((c * ((OC * 3u) * 3u)) + ((oc * (3u * 3u)) + ((kh * 3u) + kw)))]));
      }
    }
  }
  output[(oc + (oh * (OW * OC)))] = sum;
}