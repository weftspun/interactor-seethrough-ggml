groupshared float sQ[512];
groupshared float sK[512];
groupshared float sAttn[64];

[[vk::binding(0, 0)]]
StructuredBuffer<float> Q;
[[vk::binding(1, 0)]]
StructuredBuffer<float> K;
[[vk::binding(2, 0)]]
StructuredBuffer<float> V;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> O;
[[vk::binding(4, 0)]]
uint T;
[[vk::binding(5, 0)]]
uint d;

[shader("compute")] [numthreads(8, 8, 1)]
void attention(uint3 dtid : SV_DispatchThreadID) {
  uint row = dtid[0u];
  uint col = dtid[1u];
  float sum = 0.000000;
  for (uint k = 0u; k < d; ++k) {
    sum = (sum + (Q[((row * d) + k)] * K[((col * d) + k)]));
  }
  sAttn[(row + (col * 8u))] = (sum / sqrt(d));
  GroupMemoryBarrierWithGroupSync();
  float rowMax = sAttn[row];
  for (uint j = 1u; j < T; ++j) {
    rowMax = max(rowMax, sAttn[(row + (j * 8u))]);
  }
  float rowSum = 0.000000;
  for (uint j = 0u; j < T; ++j) {
    sAttn[(row + (j * 8u))] = exp(((row + (j * 8u)) - rowMax));
    rowSum = (rowSum + sAttn[(row + (j * 8u))]);
  }
  for (uint j = 0u; j < T; ++j) {
    sAttn[(row + (j * 8u))] = (sAttn[(row + (j * 8u))] / rowSum);
  }
  GroupMemoryBarrierWithGroupSync();
  sum = 0.000000;
  for (uint j = 0u; j < T; ++j) {
    sum = (sum + (sAttn[(row + (j * 8u))] * V[((j * d) + col)]));
  }
  O[((row * d) + col)] = sum;
}