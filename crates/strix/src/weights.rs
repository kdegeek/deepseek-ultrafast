//! Header-only safetensors catalog, bounded read_exact, and a Vulkan weight LRU.
//! No checkpoint mappings or full-model allocation. Files stay open read-only.
use crate::{
    math,
    reference::{self, Format},
    runtime::{Gpu, Matrix},
};
use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, tensor::Metadata};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};

const MAX_HEADER: u64 = 32 * 1024 * 1024;
pub const MAX_READ: usize = 64 * 1024 * 1024;
const CHUNK_ROWS: usize = 2048;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    file: usize,
    offset: u64,
    bytes: usize,
}

pub struct Store {
    files: Vec<File>,
    pub tensors: BTreeMap<String, Tensor>,
    pub read_bytes: u64,
    pub peak_read_bytes: usize,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        let index: Index =
            serde_json::from_reader(File::open(dir.join("model.safetensors.index.json"))?)?;
        ensure!(!index.weight_map.is_empty(), "empty checkpoint index");
        let mut names: Vec<_> = index.weight_map.values().cloned().collect();
        names.sort();
        names.dedup();
        let mut store = Self {
            files: Vec::new(),
            tensors: BTreeMap::new(),
            read_bytes: 0,
            peak_read_bytes: 0,
        };
        for name in names {
            ensure!(
                !name.contains(['/', '\\', ':']) && name.ends_with(".safetensors"),
                "invalid shard path"
            );
            let mut options = std::fs::OpenOptions::new();
            options.read(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                options.share_mode(1); // FILE_SHARE_READ only: prevent write/delete while open.
            }
            let mut file = options
                .open(dir.join(&name))
                .with_context(|| format!("open {name}"))?;
            let size = file.metadata()?.len();
            let mut length = [0; 8];
            file.read_exact(&mut length)?;
            let length = u64::from_le_bytes(length);
            ensure!(
                length > 0 && length <= MAX_HEADER && length + 8 <= size,
                "invalid header size in {name}"
            );
            let mut header = vec![0; length as usize];
            file.read_exact(&mut header)?;
            // Metadata deserialization validates shape sizes, overflow, and contiguous offsets.
            let metadata: Metadata =
                serde_json::from_slice(&header).with_context(|| format!("header {name}"))?;
            let mut end = 0;
            for (key, info) in metadata.tensors() {
                ensure!(
                    index.weight_map.get(&key) == Some(&name),
                    "index mismatch for {key}"
                );
                let (start, stop) = info.data_offsets;
                end = end.max(stop as u64);
                let tensor = Tensor {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    file: store.files.len(),
                    offset: length + 8 + start as u64,
                    bytes: stop - start,
                };
                ensure!(
                    store.tensors.insert(key.clone(), tensor).is_none(),
                    "duplicate {key}"
                );
            }
            ensure!(
                end + length + 8 == size,
                "truncated or trailing shard payload {name}"
            );
            store.files.push(file);
        }
        ensure!(
            store.tensors.len() == index.weight_map.len(),
            "missing indexed tensors"
        );
        Ok(store)
    }

    pub fn info(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn expect(&self, name: &str, shape: &[usize], dtype: Dtype) -> Result<()> {
        let t = self.info(name)?;
        ensure!(
            t.shape == shape && t.dtype == dtype,
            "{name}: expected {dtype:?}{shape:?}, found {:?}{:?}",
            t.dtype,
            t.shape
        );
        Ok(())
    }

    pub fn bytes(&mut self, name: &str, offset: usize, size: usize) -> Result<Vec<u8>> {
        let tensor = self.info(name)?.clone();
        ensure!(
            size <= MAX_READ
                && offset
                    .checked_add(size)
                    .is_some_and(|end| end <= tensor.bytes),
            "{name}: read outside bounded tensor range"
        );
        let file = &mut self.files[tensor.file];
        file.seek(SeekFrom::Start(tensor.offset + offset as u64))?;
        let mut bytes = vec![0; size];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read {name}"))?;
        self.read_bytes += size as u64;
        self.peak_read_bytes = self.peak_read_bytes.max(size);
        Ok(bytes)
    }

    pub fn small(&mut self, name: &str) -> Result<Vec<f32>> {
        let t = self.info(name)?.clone();
        ensure!(t.bytes <= 1024 * 1024, "{name}: not a small host tensor");
        let bytes = self.bytes(name, 0, t.bytes)?;
        dense_values(t.dtype, &bytes)
    }

    pub fn row(&mut self, name: &str, row: usize) -> Result<Vec<f32>> {
        let t = self.info(name)?.clone();
        ensure!(
            t.shape.len() == 2 && row < t.shape[0],
            "invalid row for {name}"
        );
        let stride = t.bytes / t.shape[0];
        let bytes = self.bytes(name, row * stride, stride)?;
        if t.dtype == Dtype::F8_E4M3 {
            let scale_name = name
                .strip_suffix("weight")
                .context("embedding name")?
                .to_owned()
                + "scale";
            let scale = self.info(&scale_name)?.clone();
            ensure!(
                scale.shape == [t.shape[0], t.shape[1] / 32] && scale.dtype == Dtype::F8_E8M0,
                "embedding scale layout"
            );
            let scales = self.bytes(&scale_name, row * t.shape[1] / 32, t.shape[1] / 32)?;
            let values: Vec<_> = bytes
                .iter()
                .enumerate()
                .map(|(i, &v)| math::bf16(reference::decode_e4m3(v) * math::e8m0(scales[i / 32])))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite embedding row"
            );
            Ok(values)
        } else {
            dense_values(t.dtype, &bytes)
        }
    }
}

fn dense_values(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    let values: Vec<_> = match dtype {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| half::bf16::from_le_bytes(*v).to_f32())
            .collect(),
        _ => bail!("unsupported dense tensor dtype {dtype:?}"),
    };
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite dense tensor"
    );
    Ok(values)
}

struct Cached {
    matrix: Matrix,
    bytes: usize,
    used: u64,
}

#[derive(Default, Serialize)]
pub struct Stats {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub peak_resident_bytes: usize,
    pub uploads: u64,
    pub cache_hits: u64,
    pub matvecs: u64,
    pub upload_seconds: f64,
    pub matvec_seconds: f64,
    /// Upper bound for simultaneous owned checkpoint bytes + packed upload copy.
    pub peak_host_weight_staging_bound_bytes: usize,
}

pub struct Weights {
    #[cfg(all(feature = "native-aten", windows))]
    pub aten: Option<crate::aten::Projector>,
    pub store: Store,
    pub gpu: Option<Gpu>,
    cache: HashMap<String, Cached>,
    clock: u64,
    pub stats: Stats,
}

impl Weights {
    pub fn clear(&mut self) {
        self.cache.clear();
        #[cfg(all(feature = "native-aten", windows))]
        if let Some(aten) = &mut self.aten {
            aten.clear();
        }
        self.stats.resident_bytes = 0;
    }
    pub fn new(store: Store, gpu: Gpu, budget: usize) -> Result<Self> {
        Self::with_gpu(store, Some(gpu), budget)
    }
    #[cfg(all(feature = "native-aten", windows))]
    pub fn new_aten(
        store: Store,
        api: std::rc::Rc<crate::aten::Api>,
        budget: usize,
    ) -> Result<Self> {
        let mut weights = Self::with_gpu(store, None, budget)?;
        weights.aten = Some(crate::aten::Projector::new(api));
        Ok(weights)
    }
    fn with_gpu(store: Store, gpu: Option<Gpu>, budget: usize) -> Result<Self> {
        ensure!(
            (MAX_READ..=32usize * 1024 * 1024 * 1024).contains(&budget),
            "weight cache must be 64 MiB through 32 GiB"
        );
        Ok(Self {
            #[cfg(all(feature = "native-aten", windows))]
            aten: None,
            store,
            gpu,
            cache: HashMap::new(),
            clock: 0,
            stats: Stats {
                budget_bytes: budget,
                ..Stats::default()
            },
        })
    }

    /// Projections execute on the explicitly selected AMD backend. Quantized inputs use
    /// E4M3/E8M0, except wo_a (explicit pre-quantization and BF16 einsum).
    pub fn linear(&mut self, name: &str, input: &[f32], round_output: bool) -> Result<Vec<f32>> {
        let rows = self.store.info(name)?.shape[0];
        self.range(name, input, 0, rows, round_output)
    }

    pub fn range(
        &mut self,
        name: &str,
        input: &[f32],
        first: usize,
        rows: usize,
        round_output: bool,
    ) -> Result<Vec<f32>> {
        #[cfg(all(feature = "native-aten", windows))]
        if let Some(aten) = &mut self.aten {
            return aten.range(
                &mut self.store,
                &mut self.stats,
                name,
                input,
                first,
                rows,
                round_output,
            );
        }
        let gpu = self
            .gpu
            .as_ref()
            .context("Vulkan backend was not selected")?;
        let tensor = self.store.info(name)?.clone();
        ensure!(
            tensor.shape.len() == 2
                && first
                    .checked_add(rows)
                    .is_some_and(|n| n <= tensor.shape[0])
                && rows > 0,
            "invalid matrix range {name}"
        );
        let format = match tensor.dtype {
            Dtype::F32 => Format::F32,
            Dtype::BF16 => Format::Bf16,
            Dtype::F8_E4M3 => Format::Fp8,
            Dtype::I8 if name.contains(".ffn.experts.") => Format::Fp4,
            other => bail!("unsupported matrix dtype {other:?} for {name}"),
        };
        let cols = tensor.shape[1] * if format == Format::Fp4 { 2 } else { 1 };
        ensure!(
            input.len() == cols,
            "{name}: input {} != {cols}",
            input.len()
        );
        let mut x = input.to_vec();
        if matches!(format, Format::Fp8 | Format::Fp4) && !name.ends_with(".wo_a.weight") {
            math::quant_fp8(&mut x)?;
        }
        let stride = tensor.bytes / tensor.shape[0];
        let mut output = Vec::with_capacity(rows);
        for start in (first..first + rows).step_by(CHUNK_ROWS) {
            let count = CHUNK_ROWS.min(first + rows - start);
            let key = format!("{name}/{start}/{count}");
            self.clock += 1;
            if !self.cache.contains_key(&key) {
                let timer = Instant::now();
                let mut scales = Vec::new();
                if matches!(format, Format::Fp8 | Format::Fp4) {
                    ensure!(
                        start.is_multiple_of(32) && cols.is_multiple_of(32),
                        "unaligned quantized matrix range"
                    );
                    let scale_name = name.strip_suffix("weight").unwrap().to_owned() + "scale";
                    let scale_rows = if format == Format::Fp4 {
                        tensor.shape[0]
                    } else {
                        tensor.shape[0].div_ceil(32)
                    };
                    self.store
                        .expect(&scale_name, &[scale_rows, cols / 32], Dtype::F8_E8M0)?;
                    let divisor = if format == Format::Fp4 { 1 } else { 32 };
                    scales = self.store.bytes(
                        &scale_name,
                        start / divisor * (cols / 32),
                        count.div_ceil(divisor) * (cols / 32),
                    )?;
                }
                let size = (count * stride).div_ceil(4) * 4 + scales.len().div_ceil(4).max(1) * 4;
                ensure!(
                    size <= self.stats.budget_bytes,
                    "matrix exceeds weight budget"
                );
                while self.stats.resident_bytes + size > self.stats.budget_bytes {
                    let victim = self
                        .cache
                        .iter()
                        .min_by_key(|(_, v)| v.used)
                        .map(|(k, _)| k.clone())
                        .context("weight cache cannot evict")?;
                    let old = self.cache.remove(&victim).unwrap();
                    self.stats.resident_bytes -= old.bytes;
                    drop(old); // last readback has waited for completion; no active weight pins.
                }
                let data = self.store.bytes(name, start * stride, count * stride)?;
                let weights = reference::pack_bytes(&data);
                let packed_scales = reference::pack_bytes(&scales);
                self.stats.peak_host_weight_staging_bound_bytes = self
                    .stats
                    .peak_host_weight_staging_bound_bytes
                    .max(2 * size);
                let matrix = gpu.upload_matrix(&weights, &packed_scales, count, cols, format)?;
                self.cache.insert(
                    key.clone(),
                    Cached {
                        matrix,
                        bytes: size,
                        used: self.clock,
                    },
                );
                self.stats.resident_bytes += size;
                self.stats.peak_resident_bytes = self
                    .stats
                    .peak_resident_bytes
                    .max(self.stats.resident_bytes);
                self.stats.uploads += 1;
                self.stats.upload_seconds += timer.elapsed().as_secs_f64();
            } else {
                self.stats.cache_hits += 1;
            }
            let cached = self.cache.get_mut(&key).unwrap();
            cached.used = self.clock;
            let timer = Instant::now();
            output.extend(
                gpu.matvec(&cached.matrix, &x)
                    .with_context(|| format!("Vulkan {key}"))?,
            );
            self.stats.matvec_seconds += timer.elapsed().as_secs_f64();
            self.stats.matvecs += 1;
        }
        if round_output {
            math::round_bf16(&mut output);
        }
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "nonfinite projection {name}"
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    struct Fixture(PathBuf);
    use std::path::PathBuf;
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "strix-read-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn write(&self, header: Value, data: &[u8]) {
            let header = serde_json::to_vec(&header).unwrap();
            let mut f = File::create(self.0.join("part.safetensors")).unwrap();
            f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&header).unwrap();
            f.write_all(data).unwrap();
            std::fs::write(
                self.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&json!({"weight_map":{"x":"part.safetensors"}})).unwrap(),
            )
            .unwrap();
        }
    }
    use serde_json::Value;
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn bounded_reader_and_immutable_handle() -> Result<()> {
        let f = Fixture::new();
        f.write(
            json!({"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}),
            bytemuck::cast_slice(&[1f32, 2., 3., 4.]),
        );
        let mut s = Store::open(&f.0)?;
        assert_eq!(s.row("x", 1)?, vec![3., 4.]);
        assert_eq!(s.read_bytes, 8);
        assert!(s.row("x", 2).is_err());
        assert!(s.bytes("x", 15, 2).is_err());
        assert!(s.bytes("x", 0, MAX_READ + 1).is_err());
        #[cfg(windows)]
        assert!(
            std::fs::OpenOptions::new()
                .write(true)
                .open(f.0.join("part.safetensors"))
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn rejects_header_shape_payload_and_path_errors() {
        let f = Fixture::new();
        for header in [
            json!({"x":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}),
            json!({"x":{"dtype":"F32","shape":[4],"data_offsets":[4,20]}}),
            json!({"x":{"dtype":"F32","shape":[usize::MAX,4],"data_offsets":[0,16]}}),
        ] {
            f.write(header, &[0; 16]);
            assert!(Store::open(&f.0).is_err());
        }
        f.write(
            json!({"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}),
            &[0; 15],
        );
        assert!(Store::open(&f.0).is_err());
        std::fs::write(
            f.0.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"../part.safetensors"}}"#,
        )
        .unwrap();
        assert!(Store::open(&f.0).is_err());
    }
}
