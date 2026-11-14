// Native Vulkan baseline. Packed values are little-endian, row-major checkpoint
// data, before the CUDA-specific tensor-core tiling in model::formats.
struct Params {
    rows: u32,
    cols: u32,
    format: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> auxiliary: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

fn scale_at(index: u32) -> f32 {
    let code = (scales[index / 4u] >> ((index % 4u) * 8u)) & 255u;
    // Match model::formats::e8m0_to_f32, including code zero -> zero.
    return bitcast<f32>(code << 23u);
}

fn e4m3(code: u32) -> f32 {
    let exponent = (code >> 3u) & 15u;
    let mantissa = code & 7u;
    let sign = select(1.0, -1.0, (code & 128u) != 0u);
    if exponent == 0u {
        return sign * f32(mantissa) / 512.0;
    }
    // Non-finite checkpoint values are rejected by host validation.
    return sign * (1.0 + f32(mantissa) / 8.0) * exp2(f32(i32(exponent) - 7));
}

fn e2m1(code: u32) -> f32 {
    let magnitudes = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    return select(1.0, -1.0, (code & 8u) != 0u) * magnitudes[code & 7u];
}

fn weight_at(row: u32, col: u32) -> f32 {
    let index = row * params.cols + col;
    switch params.format {
        case 0u: { return bitcast<f32>(weights[index]); }
        case 1u: {
            let code = (weights[index / 2u] >> ((index % 2u) * 16u)) & 65535u;
            return bitcast<f32>(code << 16u);
        }
        case 2u: {
            let code = (weights[index / 4u] >> ((index % 4u) * 8u)) & 255u;
            let scale_index = (row / 32u) * (params.cols / 32u) + col / 32u;
            return e4m3(code) * scale_at(scale_index);
        }
        case 3u: {
            let code = (weights[index / 8u] >> ((index % 8u) * 4u)) & 15u;
            let scale_index = row * (params.cols / 32u) + col / 32u;
            return e2m1(code) * scale_at(scale_index);
        }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(64)
fn matvec(@builtin(workgroup_id) group: vec3<u32>,
          @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    if row >= params.rows { return; }
    var sum = 0.0;
    for (var col = lane; col < params.cols; col += 64u) {
        sum += weight_at(row, col) * input[col];
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { output[row] = partial[0]; }
}

@compute @workgroup_size(64)
fn rms_norm(@builtin(workgroup_id) group: vec3<u32>,
            @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    if row >= params.rows { return; }
    var sum = 0.0;
    for (var col = lane; col < params.cols; col += 64u) {
        let x = input[row * params.cols + col];
        sum += x * x;
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    let inv_rms = inverseSqrt(partial[0] / f32(params.cols) + params.epsilon);
    for (var col = lane; col < params.cols; col += 64u) {
        let index = row * params.cols + col;
        output[index] = input[index] * inv_rms * auxiliary[col];
    }
}

@compute @workgroup_size(64)
fn silu_mul(@builtin(global_invocation_id) global: vec3<u32>) {
    let index = global.x;
    if index >= params.rows * params.cols { return; }
    let x = input[index];
    var silu: f32;
    if x >= 0.0 {
        silu = x / (1.0 + exp(-x));
    } else {
        let ex = exp(x);
        silu = x * ex / (1.0 + ex);
    }
    output[index] = silu * auxiliary[index];
}
