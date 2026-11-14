//! Checkpoint inventory and bounded, untiled matrix extraction.
//!
//! This reads safetensors storage layouts, not the CUDA-specific tiled layouts.
//! Files must remain unchanged for the duration of a call (they are memory mapped).
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Component, Path};

/// Cap total copied weight and scale bytes for a single diagnostic matrix.
pub const MAX_MATRIX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Default, Serialize)]
pub struct DtypeTotal {
    pub tensor_count: usize,
    pub stored_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct ShardReport {
    pub name: String,
    pub tensor_count: usize,
    pub stored_bytes: u64,
    pub file_bytes: u64,
}

/// Storage accounting only: runtime activations, KV cache, and staging are excluded.
#[derive(Debug, Serialize)]
pub struct Report {
    pub tensor_count: usize,
    pub stored_bytes: u64,
    pub file_bytes: u64,
    pub dtype_totals: BTreeMap<String, DtypeTotal>,
    pub shards: Vec<ShardReport>,
}

#[derive(Debug)]
pub struct OwnedMatrix {
    pub rows: usize,
    pub cols: usize,
    /// 0: F32; 1: BF16; 2: E4M3 with 32x32 scales; 3: E2M1 with 1x32 scales.
    pub format: u32,
    pub weights: Vec<u8>,
    pub scales: Vec<u8>,
}

#[derive(Debug)]
struct Location {
    shard: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: usize,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
    #[serde(default)]
    metadata: IndexMetadata,
}

#[derive(Default, Deserialize)]
struct IndexMetadata {
    total_size: Option<u64>,
}

struct Catalog {
    report: Report,
    tensors: BTreeMap<String, Location>,
}

fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: read-only mapping; callers must not modify/truncate checkpoint files
    // during inspection or extraction. No references escape this module's calls.
    unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", path.display()))
}

fn shard_filename(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(Component::Normal(_)))
        && parts.next().is_none()
        && !name.contains(['/', '\\', ':'])
        && name.ends_with(".safetensors")
}

fn catalog(dir: &Path) -> Result<Catalog> {
    ensure!(
        dir.is_dir(),
        "checkpoint directory {} does not exist",
        dir.display()
    );
    let mut files = Vec::new();
    let mut indexes = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 checkpoint filename"))?;
        if name.ends_with(".safetensors") {
            files.push(name);
        } else if name.ends_with(".safetensors.index.json") {
            indexes.push(entry.path());
        }
    }
    files.sort();
    ensure!(
        !files.is_empty(),
        "no safetensors shards in {}",
        dir.display()
    );
    ensure!(
        indexes.len() <= 1,
        "multiple safetensors indexes in {}; select one checkpoint directory",
        dir.display()
    );
    let index: Option<Index> = indexes
        .first()
        .map(|path| -> Result<Index> {
            serde_json::from_reader(File::open(path)?)
                .with_context(|| format!("parse {}", path.display()))
        })
        .transpose()?;
    if let Some(index) = &index {
        ensure!(
            !index.weight_map.is_empty(),
            "checkpoint index has an empty weight_map"
        );
        for (tensor, shard) in &index.weight_map {
            ensure!(
                shard_filename(shard),
                "invalid shard filename {shard:?} for {tensor}"
            );
            ensure!(
                files.binary_search(shard).is_ok(),
                "missing checkpoint shard {shard} referenced by {tensor}"
            );
        }
    }
    let mut result = Catalog {
        report: Report {
            tensor_count: 0,
            stored_bytes: 0,
            file_bytes: 0,
            dtype_totals: BTreeMap::new(),
            shards: Vec::new(),
        },
        tensors: BTreeMap::new(),
    };
    for name in files {
        let map = map_file(&dir.join(&name))?;
        let tensors = SafeTensors::deserialize(&map)
            .with_context(|| format!("invalid safetensors shard {name}"))?;
        let mut shard = ShardReport {
            name: name.clone(),
            tensor_count: tensors.len(),
            stored_bytes: 0,
            file_bytes: map.len() as u64,
        };
        for (tensor_name, view) in tensors.iter() {
            ensure!(
                !result.tensors.contains_key(tensor_name),
                "duplicate tensor {tensor_name} across shards"
            );
            if let Some(index) = &index {
                ensure!(
                    index.weight_map.get(tensor_name) == Some(&name),
                    "tensor {tensor_name} in {name} does not match checkpoint index"
                );
            }
            let bytes = view.data().len();
            shard.stored_bytes = shard
                .stored_bytes
                .checked_add(bytes as u64)
                .context("checkpoint size overflow")?;
            let totals = result
                .report
                .dtype_totals
                .entry(format!("{:?}", view.dtype()))
                .or_default();
            totals.tensor_count += 1;
            totals.stored_bytes = totals
                .stored_bytes
                .checked_add(bytes as u64)
                .context("dtype size overflow")?;
            result.tensors.insert(
                tensor_name.to_string(),
                Location {
                    shard: name.clone(),
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes,
                },
            );
        }
        result.report.stored_bytes = result
            .report
            .stored_bytes
            .checked_add(shard.stored_bytes)
            .context("checkpoint size overflow")?;
        result.report.file_bytes = result
            .report
            .file_bytes
            .checked_add(shard.file_bytes)
            .context("checkpoint file size overflow")?;
        result.report.shards.push(shard);
    }
    result.report.tensor_count = result.tensors.len();
    ensure!(
        result.report.tensor_count > 0,
        "checkpoint contains no tensors"
    );
    if let Some(index) = index {
        for tensor in index.weight_map.keys() {
            ensure!(
                result.tensors.contains_key(tensor),
                "indexed tensor {tensor} is missing from checkpoint"
            );
        }
        if let Some(expected) = index.metadata.total_size {
            ensure!(
                expected == result.report.stored_bytes,
                "index total_size {expected} differs from actual tensor bytes {}",
                result.report.stored_bytes
            );
        }
    }
    Ok(result)
}

pub fn inspect(dir: &Path) -> Result<Report> {
    Ok(catalog(dir)?.report)
}

fn copy_tensor(dir: &Path, name: &str, location: &Location) -> Result<Vec<u8>> {
    let map = map_file(&dir.join(&location.shard))?;
    let tensors = SafeTensors::deserialize(&map)?;
    let tensor = tensors.tensor(name)?;
    ensure!(
        tensor.dtype() == location.dtype
            && tensor.shape() == location.shape
            && tensor.data().len() == location.bytes,
        "tensor {name} changed during extraction"
    );
    Ok(tensor.data().to_vec())
}

fn check_matrix_budget(weight_bytes: usize, scale_bytes: usize) -> Result<()> {
    let bytes = weight_bytes
        .checked_add(scale_bytes)
        .context("matrix byte count overflow")?;
    ensure!(
        bytes <= MAX_MATRIX_BYTES,
        "matrix needs {bytes} bytes; diagnostic extraction limit is {MAX_MATRIX_BYTES}"
    );
    Ok(())
}

#[derive(Deserialize)]
struct ReleaseConfig {
    model_type: String,
    quantization_config: ReleaseQuantization,
    text_config: ReleaseTextConfig,
}

#[derive(Deserialize)]
struct ReleaseQuantization {
    quant_method: String,
    expert_dtype: String,
    scale_fmt: String,
    weight_block_size: [usize; 2],
}

#[derive(Deserialize)]
struct ReleaseTextConfig {
    hidden_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    n_routed_experts: usize,
    #[serde(default)]
    num_nextn_predict_layers: usize,
    #[serde(default)]
    dspark_n_routed_experts: usize,
}

/// The released model stores two FP4 values in each I8 element. Authoritative
/// convention: DeepSeek-V4.1-Flash/inference/convert.py (cast_e2m1fn_to_e4m3fn)
/// and inference/model.py (Linear). Do not reinterpret arbitrary integer tensors.
fn packed_expert_cols(dir: &Path, name: &str, rows: usize, stored_cols: usize) -> Result<usize> {
    let config: ReleaseConfig = serde_json::from_reader(
        File::open(dir.join("config.json"))
            .context("packed I8 experts require the released config.json")?,
    )
    .context("read packed expert model configuration")?;
    let quant = config.quantization_config;
    ensure!(
        config.model_type == "deepseek_v41"
            && quant.quant_method == "fp8"
            && quant.expert_dtype == "fp4"
            && quant.scale_fmt == "ue8m0"
            && quant.weight_block_size == [32, 32],
        "I8 reinterpretation requires DeepSeek V4.1 FP4 expert / UE8M0 configuration"
    );
    let parts: Vec<_> = name
        .strip_prefix("model.")
        .unwrap_or(name)
        .split('.')
        .collect();
    ensure!(
        parts.len() == 7
            && matches!(parts[0], "layers" | "mtp")
            && matches!(parts[2], "ffn" | "mlp")
            && parts[3] == "experts"
            && matches!(parts[5], "w1" | "w2" | "w3")
            && parts[6] == "weight",
        "{name}: I8 reinterpretation is restricted to routed expert w1/w2/w3 weights"
    );
    let layer: usize = parts[1].parse().context("invalid expert layer index")?;
    let expert: usize = parts[4].parse().context("invalid expert index")?;
    let text = config.text_config;
    let (layer_count, expert_count) = if parts[0] == "mtp" {
        (text.num_nextn_predict_layers, text.dspark_n_routed_experts)
    } else {
        (text.num_hidden_layers, text.n_routed_experts)
    };
    ensure!(
        layer < layer_count && expert < expert_count,
        "{name}: expert index outside configured model"
    );
    let cols = stored_cols
        .checked_mul(2)
        .context("packed FP4 dimension overflow")?;
    let expected = if parts[5] == "w2" {
        (text.hidden_size, text.moe_intermediate_size)
    } else {
        (text.moe_intermediate_size, text.hidden_size)
    };
    ensure!(
        (rows, cols) == expected,
        "{name}: packed I8 matrix expands to {rows}x{cols}, expected {}x{} from config",
        expected.0,
        expected.1
    );
    Ok(cols)
}

/// Extract one explicitly named matrix. FP4 may declare safetensors `F4`, or
/// use the released model's I8 expert packing when config, name and shape agree.
/// Ambiguous byte-packed `U8` weights are rejected rather than guessed.
pub fn load_matrix(dir: &Path, name: &str) -> Result<OwnedMatrix> {
    let catalog = catalog(dir)?;
    let weight = catalog
        .tensors
        .get(name)
        .with_context(|| format!("tensor {name} not found"))?;
    ensure!(
        weight.shape.len() == 2,
        "{name}: expected a 2D matrix, found {:?}",
        weight.shape
    );
    let (rows, stored_cols) = (weight.shape[0], weight.shape[1]);
    let cols = if weight.dtype == Dtype::I8 {
        packed_expert_cols(dir, name, rows, stored_cols)?
    } else {
        stored_cols
    };
    ensure!(
        rows > 0 && cols > 0,
        "{name}: empty matrices are unsupported"
    );
    rows.checked_mul(cols)
        .context("matrix element count overflow")?;
    let format = match weight.dtype {
        Dtype::F32 => 0,
        Dtype::BF16 => 1,
        Dtype::F8_E4M3 => 2,
        Dtype::F4 | Dtype::I8 => 3,
        dtype => {
            bail!("{name}: unsupported weight dtype {dtype:?}; packed U8 layouts are not inferred")
        }
    };
    let mut scale_location = None;
    let mut scale_name = String::new();
    if format >= 2 {
        ensure!(
            cols.is_multiple_of(32),
            "{name}: quantized columns must be a multiple of 32"
        );
        if format == 2 {
            ensure!(
                rows.is_multiple_of(32),
                "{name}: FP8 rows must be a multiple of 32"
            );
        }
        let prefix = name
            .strip_suffix(".weight")
            .context("quantized matrix name must end in .weight")?;
        scale_name = format!("{prefix}.scale");
        let scale = catalog
            .tensors
            .get(&scale_name)
            .with_context(|| format!("missing scale tensor {scale_name}"))?;
        ensure!(
            matches!(scale.dtype, Dtype::U8 | Dtype::F8_E8M0),
            "{scale_name}: expected U8/E8M0 scales, found {:?}",
            scale.dtype
        );
        if weight.dtype == Dtype::I8 {
            ensure!(
                scale.dtype == Dtype::F8_E8M0,
                "{scale_name}: released packed I8 experts require typed F8_E8M0 scales"
            );
        }
        let expected_shape = [if format == 2 { rows / 32 } else { rows }, cols / 32];
        ensure!(
            scale.shape == expected_shape,
            "{scale_name}: expected shape {expected_shape:?}, found {:?}",
            scale.shape
        );
        scale_location = Some(scale);
    }
    check_matrix_budget(weight.bytes, scale_location.map_or(0, |s| s.bytes))
        .with_context(|| format!("{name}: extraction refused"))?;
    let weights = copy_tensor(dir, name, weight)?;
    let scales = scale_location
        .map(|s| copy_tensor(dir, &scale_name, s))
        .transpose()?
        .unwrap_or_default();
    ensure!(
        !scales.contains(&255),
        "{scale_name}: non-finite E8M0 scale code 255 is unsupported"
    );
    Ok(OwnedMatrix {
        rows,
        cols,
        format,
        weights,
        scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "dsuf-strix-checkpoint-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn shard(&self, name: &str, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
            let views: Vec<_> = tensors
                .iter()
                .map(|(name, dtype, shape, bytes)| {
                    (
                        *name,
                        safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect();
            let data = safetensors::serialize(views, None).unwrap();
            std::fs::write(self.0.join(name), data).unwrap();
        }
        fn index(&self, value: serde_json::Value) {
            std::fs::write(
                self.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn budget_includes_scales_and_rejects_overflow() {
        assert!(check_matrix_budget(MAX_MATRIX_BYTES - 4, 4).is_ok());
        assert!(check_matrix_budget(MAX_MATRIX_BYTES, 1).is_err());
        assert!(check_matrix_budget(usize::MAX, 1).is_err());
    }

    fn release_config() -> serde_json::Value {
        json!({
            "model_type":"deepseek_v41",
            "quantization_config":{"quant_method":"fp8","expert_dtype":"fp4",
                "scale_fmt":"ue8m0","weight_block_size":[32,32]},
            "text_config":{"hidden_size":5120,"moe_intermediate_size":2304,
                "num_hidden_layers":40,"n_routed_experts":384,
                "num_nextn_predict_layers":3,"dspark_n_routed_experts":128}
        })
    }

    #[test]
    fn extracts_released_i8_expert_layout_with_verified_configuration() {
        let fixture = Fixture::new();
        // Exact metadata checked against the released shard 00009 header:
        // logical 2304x5120, I8 storage 2304x2560, E8M0 scales 2304x160.
        let name = "layers.6.ffn.experts.0.w1.weight";
        fixture.shard(
            "model.safetensors",
            &[
                (name, Dtype::I8, vec![2304, 2560], vec![0x92; 2304 * 2560]),
                (
                    "layers.6.ffn.experts.0.w1.scale",
                    Dtype::F8_E8M0,
                    vec![2304, 160],
                    vec![127; 2304 * 160],
                ),
            ],
        );
        assert!(load_matrix(&fixture.0, name).is_err());
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&release_config()).unwrap(),
        )
        .unwrap();
        let matrix = load_matrix(&fixture.0, name).unwrap();
        assert_eq!((matrix.rows, matrix.cols, matrix.format), (2304, 5120, 3));
        assert_eq!(matrix.weights.len(), 2304 * 2560);
        assert_eq!(matrix.weights[0], 0x92);
        assert_eq!(matrix.scales.len(), 2304 * 160);
        assert_eq!(
            packed_expert_cols(&fixture.0, "layers.6.ffn.experts.0.w2.weight", 5120, 1152).unwrap(),
            2304
        );
        assert_eq!(
            packed_expert_cols(&fixture.0, "mtp.2.ffn.experts.127.w3.weight", 2304, 2560).unwrap(),
            5120
        );
        assert!(
            packed_expert_cols(&fixture.0, "mtp.2.ffn.experts.128.w3.weight", 2304, 2560).is_err()
        );
        assert!(
            packed_expert_cols(&fixture.0, "layers.40.ffn.experts.0.w1.weight", 2304, 2560)
                .is_err()
        );
        assert!(
            packed_expert_cols(
                &fixture.0,
                "layers.6.ffn.shared_experts.w1.weight",
                2304,
                2560
            )
            .is_err()
        );
        assert!(packed_expert_cols(&fixture.0, name, 2304, 5120).is_err());
        let mut wrong = release_config();
        wrong["quantization_config"]["expert_dtype"] = json!("int8");
        std::fs::write(
            fixture.0.join("config.json"),
            serde_json::to_vec(&wrong).unwrap(),
        )
        .unwrap();
        assert!(load_matrix(&fixture.0, name).is_err());
    }

    #[test]
    fn reports_storage_and_extracts_dense_matrix() {
        let fixture = Fixture::new();
        let data = [1.0f32, -2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        fixture.shard(
            "model.safetensors",
            &[("a.weight", Dtype::F32, vec![2, 2], data.clone())],
        );
        fixture.index(
            json!({"metadata":{"total_size":16}, "weight_map":{"a.weight":"model.safetensors"}}),
        );
        let report = inspect(&fixture.0).unwrap();
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.stored_bytes, 16);
        assert!(report.file_bytes > report.stored_bytes);
        assert_eq!(report.dtype_totals["F32"].stored_bytes, 16);
        let matrix = load_matrix(&fixture.0, "a.weight").unwrap();
        assert_eq!((matrix.rows, matrix.cols, matrix.format), (2, 2, 0));
        assert_eq!(matrix.weights, data);
        assert!(matrix.scales.is_empty());
    }

    #[test]
    fn extracts_fp4_and_fp8_with_distinct_scale_shapes() {
        let fixture = Fixture::new();
        fixture.shard(
            "model.safetensors",
            &[
                ("four.weight", Dtype::F4, vec![2, 64], vec![0x12; 64]),
                ("four.scale", Dtype::F8_E8M0, vec![2, 2], vec![127; 4]),
                (
                    "eight.weight",
                    Dtype::F8_E4M3,
                    vec![32, 64],
                    vec![0x38; 2048],
                ),
                ("eight.scale", Dtype::U8, vec![1, 2], vec![128; 2]),
            ],
        );
        let four = load_matrix(&fixture.0, "four.weight").unwrap();
        assert_eq!((four.rows, four.cols, four.format), (2, 64, 3));
        assert_eq!(four.weights.len(), 64);
        let eight = load_matrix(&fixture.0, "eight.weight").unwrap();
        assert_eq!(eight.format, 2);
        assert_eq!(eight.scales, vec![128; 2]);
    }

    #[test]
    fn rejects_missing_shards_and_incorrect_index_entries() {
        let fixture = Fixture::new();
        fixture.shard(
            "one.safetensors",
            &[("a", Dtype::BF16, vec![1, 2], vec![0; 4])],
        );
        fixture.index(json!({"weight_map":{"a":"absent.safetensors"}}));
        assert!(
            inspect(&fixture.0)
                .unwrap_err()
                .to_string()
                .contains("missing checkpoint shard")
        );
        fixture.index(json!({"weight_map":{"a":"../one.safetensors"}}));
        assert!(
            inspect(&fixture.0)
                .unwrap_err()
                .to_string()
                .contains("invalid shard filename")
        );
        fixture.index(json!({"weight_map":{"a":"one.safetensors","missing":"one.safetensors"}}));
        assert!(
            inspect(&fixture.0)
                .unwrap_err()
                .to_string()
                .contains("indexed tensor missing")
        );
        fixture.index(json!({"metadata":{"total_size":5},"weight_map":{"a":"one.safetensors"}}));
        assert!(
            inspect(&fixture.0)
                .unwrap_err()
                .to_string()
                .contains("total_size")
        );
    }

    #[test]
    fn rejects_duplicate_tensors_across_shards() {
        let fixture = Fixture::new();
        for name in ["one.safetensors", "two.safetensors"] {
            fixture.shard(name, &[("a", Dtype::BF16, vec![1, 2], vec![0; 4])]);
        }
        assert!(
            inspect(&fixture.0)
                .unwrap_err()
                .to_string()
                .contains("duplicate tensor")
        );
    }

    #[test]
    fn rejects_malformed_header_and_tensor_size() {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join("bad.safetensors"), b"short").unwrap();
        assert!(inspect(&fixture.0).is_err());
        let header =
            serde_json::to_vec(&json!({"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,4]}}))
                .unwrap();
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend(header);
        data.extend([0u8; 4]);
        std::fs::write(fixture.0.join("bad.safetensors"), data).unwrap();
        assert!(inspect(&fixture.0).is_err());
    }

    #[test]
    fn rejects_ambiguous_packing_and_wrong_scales() {
        let fixture = Fixture::new();
        fixture.shard(
            "model.safetensors",
            &[("raw.weight", Dtype::U8, vec![1, 16], vec![0; 16])],
        );
        assert!(
            load_matrix(&fixture.0, "raw.weight")
                .unwrap_err()
                .to_string()
                .contains("unsupported weight dtype")
        );
        fixture.shard(
            "model.safetensors",
            &[
                ("x.weight", Dtype::F4, vec![2, 32], vec![0; 32]),
                ("x.scale", Dtype::U8, vec![1, 2], vec![127; 2]),
            ],
        );
        assert!(
            load_matrix(&fixture.0, "x.weight")
                .unwrap_err()
                .to_string()
                .contains("expected shape")
        );
        fixture.shard(
            "model.safetensors",
            &[
                ("x.weight", Dtype::F4, vec![2, 32], vec![0; 32]),
                ("x.scale", Dtype::U8, vec![2, 1], vec![255; 2]),
            ],
        );
        assert!(
            load_matrix(&fixture.0, "x.weight")
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
    }
}
