[[vk::binding(0, 0)]]
StructuredBuffer<float> x;
[[vk::binding(1, 0)]]
StructuredBuffer<float> gamma;
[[vk::binding(2, 0)]]
StructuredBuffer<float> beta;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> y;
[[vk::binding(4, 0)]]
uint C;

[shader("compute")] [numthreads(32, 1, 1)]
void layer_norm(uint3 dtid : SV_DispatchThreadID) {
  uint tid = dtid[0u];
  float mean = 0.000000;
  for (uint i = 0u; i < C; ++i) {
    mean = (mean + x[((tid * C) + i)]);
  }
  mean = (mean / C);
  float variance = 0.000000;
  for (uint i = 0u; i < C; ++i) {
    variance = (variance + ((x[((tid * C) + i)] - mean) * (x[((tid * C) + i)] - mean)));
  }
  variance = (variance / C);
  for (uint i = 0u; i < C; ++i) {
    y[((tid * C) + i)] = ((((x[((tid * C) + i)] - mean) / sqrt((variance + 0.000010))) * gamma[i]) + beta[i]);
  }
}