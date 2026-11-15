//! Explicit runtime-loaded native ATen/HIP compatibility. Windows only.
//! DLLs are intentionally retained until process exit: ATen registers process-
//! global dispatch callbacks. Tensor handles, cache entries and DLL search cookies
//! are RAII-owned. No Python, subprocess or network bridge exists here.
use crate::{math::Mix, weights::Store};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    os::windows::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr,
    rc::Rc,
};

type Module = *mut c_void;
type Handle = *mut c_void;
type Create =
    unsafe extern "C" fn(*const f32, i64, *const i64, i64, *mut Handle, *mut c_char, usize) -> i32;
type Destroy = unsafe extern "C" fn(Handle);
type Memory = unsafe extern "C" fn(*mut i64, *mut c_char, usize) -> i32;
type Bytes = unsafe extern "C" fn(*const u8, i64, i64, *mut Handle, *mut c_char, usize) -> i32;
type Copy = unsafe extern "C" fn(Handle, *mut f32, i64, *mut c_char, usize) -> i32;
type Run = unsafe extern "C" fn(
    i32,
    *const Handle,
    i64,
    *const f64,
    i64,
    *mut Handle,
    *mut c_char,
    usize,
) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn AddDllDirectory(path: *const u16) -> Module;
    fn RemoveDllDirectory(cookie: Module) -> i32;
    fn LoadLibraryExW(path: *const u16, file: Module, flags: u32) -> Module;
    fn GetModuleHandleW(name: *const u16) -> Module;
    fn GetModuleFileNameW(module: Module, path: *mut u16, size: u32) -> u32;
    fn GetProcAddress(module: Module, name: *const u8) -> *mut c_void;
    fn GetCurrentProcess() -> Module;
    fn K32EnumProcessModules(
        process: Module,
        modules: *mut Module,
        size: u32,
        needed: *mut u32,
    ) -> i32;
}
fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}
fn path_of(module: Module) -> Result<PathBuf> {
    let mut path = vec![0u16; 32768];
    let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) };
    ensure!(
        n > 0 && (n as usize) < path.len(),
        "cannot resolve loaded module"
    );
    Ok(PathBuf::from(String::from_utf16(&path[..n as usize])?))
}
fn canonical(path: &Path) -> Result<PathBuf> {
    let resolved = path
        .canonicalize()
        .with_context(|| format!("resolve {}", path.display()))?;
    // rocBLAS derives its Tensile path from the loaded DLL filename and its
    // Windows path parser rejects the verbatim \\?\ prefix. Use a normal absolute
    // local-drive path; do not alter the process/global PATH to work around it.
    let text = resolved.to_string_lossy();
    if let Some(local) = text.strip_prefix(r"\\?\") {
        ensure!(
            local.as_bytes().get(1) == Some(&b':'),
            "ATen runtime must be on a local drive"
        );
        return Ok(PathBuf::from(local));
    }
    Ok(resolved)
}
fn same_path(a: &Path, b: &Path) -> Result<bool> {
    Ok(canonical(a)?
        .to_string_lossy()
        .eq_ignore_ascii_case(&canonical(b)?.to_string_lossy()))
}
fn loaded_modules() -> Result<Vec<PathBuf>> {
    let mut modules = vec![ptr::null_mut(); 1024];
    let mut needed = 0;
    let size = std::mem::size_of_val(modules.as_slice()) as u32;
    ensure!(
        unsafe {
            K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), size, &mut needed)
        } != 0
            && needed <= size,
        "module inventory failed or exceeded bound"
    );
    modules[..needed as usize / std::mem::size_of::<Module>()]
        .iter()
        .map(|&m| path_of(m))
        .collect()
}
fn reject_python(modules: &[PathBuf]) -> Result<()> {
    ensure!(
        !modules.iter().any(|p| {
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_ascii_lowercase();
            name.starts_with("python") || name == "torch_python.dll"
        }),
        "native ATen process has a Python runtime loaded"
    );
    Ok(())
}
fn validate_module(path: &Path) -> Result<()> {
    let name = Path::new(path.file_name().context("DLL filename")?);
    let module = unsafe { GetModuleHandleW(wide(name).as_ptr()) };
    if !module.is_null() {
        let actual = path_of(module)?;
        ensure!(
            same_path(path, &actual)?,
            "DLL conflict: {} already loaded from {} (expected {})",
            name.display(),
            actual.display(),
            path.display()
        );
    }
    Ok(())
}
fn load(path: &Path) -> Result<Module> {
    validate_module(path)?;
    let module = unsafe { LoadLibraryExW(wide(path).as_ptr(), ptr::null_mut(), 0x100 | 0x1000) };
    ensure!(
        !module.is_null(),
        "load {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    ensure!(
        same_path(path, &path_of(module)?)?,
        "loaded DLL path mismatch"
    );
    Ok(module)
}
fn symbol(module: Module, name: &CStr) -> Result<*mut c_void> {
    let p = unsafe { GetProcAddress(module, name.as_ptr().cast()) };
    ensure!(
        !p.is_null(),
        "missing native symbol {}",
        name.to_string_lossy()
    );
    Ok(p)
}
struct Directory(Module);
impl Drop for Directory {
    fn drop(&mut self) {
        unsafe {
            RemoveDllDirectory(self.0);
        }
    }
}

#[derive(Clone, Serialize)]
pub struct Identity {
    pub backend: &'static str,
    pub torch_version: String,
    pub hip_runtime_version: i32,
    pub device: String,
    pub loaded_modules: Vec<PathBuf>,
}
pub struct Api {
    bytes: Bytes,
    memory: Memory,
    create: Create,
    destroy: Destroy,
    copy: Copy,
    run: Run,
    _directories: Vec<Directory>,
    runtime_directories: [PathBuf; 3],
    pub identity: Identity,
}
impl Api {
    pub fn open(site_packages: &Path, shim: &Path) -> Result<Rc<Self>> {
        let root = canonical(site_packages)?;
        let shim = canonical(shim)?;
        let paths = [
            root.join("torch/lib"),
            root.join("_rocm_sdk_core/bin"),
            root.join("_rocm_sdk_libraries/bin"),
        ];
        reject_python(&loaded_modules()?)?;
        // Reject conflicts before loading any dependency. Check all DLLs in the
        // runtime directories, not merely amdhip64, to prevent mixed SDKs.
        for dir in &paths {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("dll"))
                {
                    validate_module(&path)?;
                }
            }
        }
        let mut directories = vec![];
        for path in &paths {
            let cookie = unsafe { AddDllDirectory(wide(path).as_ptr()) };
            ensure!(
                !cookie.is_null(),
                "AddDllDirectory {} failed",
                path.display()
            );
            directories.push(Directory(cookie));
        }
        let hip = load(&paths[1].join("amdhip64_7.dll"))?;
        load(&paths[0].join("torch_hip.dll"))?;
        let module = load(&shim)?;
        let abi: unsafe extern "C" fn() -> u32 =
            unsafe { std::mem::transmute(symbol(module, c"strix_aten_abi")?) };
        ensure!(unsafe { abi() } == 1, "unsupported native ATen ABI");
        let version: unsafe extern "C" fn() -> *const c_char =
            unsafe { std::mem::transmute(symbol(module, c"strix_aten_version")?) };
        let torch_version = unsafe { CStr::from_ptr(version()) }.to_str()?.to_owned();
        ensure!(
            torch_version == "2.11.0",
            "unvalidated ATen build version {torch_version}"
        );
        let hip_version: unsafe extern "C" fn(*mut i32) -> i32 =
            unsafe { std::mem::transmute(symbol(hip, c"hipRuntimeGetVersion")?) };
        let hip_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32 =
            unsafe { std::mem::transmute(symbol(hip, c"hipDeviceGetName")?) };
        let mut hip_runtime_version = 0;
        let mut name = [0i8; 256];
        ensure!(
            unsafe { hip_version(&mut hip_runtime_version) } == 0
                && unsafe { hip_name(name.as_mut_ptr(), 256, 0) } == 0,
            "HIP runtime/device query failed"
        );
        ensure!(
            hip_runtime_version == 71460850,
            "unvalidated HIP runtime version {hip_runtime_version}"
        );
        let device = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_str()?
            .to_owned();
        ensure!(
            device.contains("Radeon") && device.contains("8060S"),
            "native compatibility requires Radeon 8060S, found {device}"
        );
        let modules = loaded_modules()?;
        reject_python(&modules)?;
        // Dependencies may have loaded transitively. Independently recheck all.
        for actual in &modules {
            for dir in &paths {
                let expected = dir.join(actual.file_name().context("DLL filename")?);
                if expected.is_file() {
                    ensure!(
                        same_path(actual, &expected)?,
                        "transitive DLL conflict: {}",
                        actual.display()
                    );
                }
            }
        }
        Ok(Rc::new(Self {
            bytes: unsafe {
                std::mem::transmute::<*mut c_void, Bytes>(symbol(module, c"strix_aten_bytes")?)
            },
            memory: unsafe {
                std::mem::transmute::<*mut c_void, Memory>(symbol(module, c"strix_aten_memory")?)
            },
            create: unsafe {
                std::mem::transmute::<*mut c_void, Create>(symbol(module, c"strix_aten_create")?)
            },
            destroy: unsafe {
                std::mem::transmute::<*mut c_void, Destroy>(symbol(module, c"strix_aten_destroy")?)
            },
            copy: unsafe {
                std::mem::transmute::<*mut c_void, Copy>(symbol(module, c"strix_aten_copy")?)
            },
            run: unsafe {
                std::mem::transmute::<*mut c_void, Run>(symbol(module, c"strix_aten_run")?)
            },
            _directories: directories,
            runtime_directories: paths,
            identity: Identity {
                backend: "experimental native ATen/HIP compatibility",
                torch_version,
                hip_runtime_version,
                device,
                loaded_modules: modules,
            },
        }))
    }
    pub fn from_env() -> Result<Rc<Self>> {
        let root = std::env::var_os("STRIX_ATEN_SITE_PACKAGES")
            .context("set STRIX_ATEN_SITE_PACKAGES explicitly")?;
        let shim = std::env::var_os("STRIX_ATEN_SHIM").context("set STRIX_ATEN_SHIM explicitly")?;
        Self::open(Path::new(&root), Path::new(&shim))
    }
    pub fn memory(&self) -> Result<serde_json::Value> {
        let mut values = [0i64; 4];
        let mut error = [0i8; 4096];
        check(
            unsafe { (self.memory)(values.as_mut_ptr(), error.as_mut_ptr(), error.len()) },
            &error,
        )?;
        Ok(
            serde_json::json!({"allocated_bytes":values[0],"peak_allocated_bytes":values[1],"reserved_bytes":values[2],"peak_reserved_bytes":values[3]}),
        )
    }
    pub fn modules(&self) -> Result<Vec<PathBuf>> {
        let modules = loaded_modules()?;
        reject_python(&modules)?;
        for actual in &modules {
            for dir in &self.runtime_directories {
                let expected = dir.join(actual.file_name().context("DLL filename")?);
                if expected.is_file() {
                    ensure!(
                        same_path(actual, &expected)?,
                        "late-loaded DLL conflict: {}",
                        actual.display()
                    );
                }
            }
        }
        Ok(modules)
    }
    pub fn tensor(self: &Rc<Self>, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .context("tensor shape overflow")?;
        ensure!(
            !data.is_empty()
                && count == data.len()
                && shape.len() <= 4
                && !shape.is_empty()
                && data.iter().all(|x| x.is_finite()),
            "invalid tensor shape/data"
        );
        let shape: Vec<i64> = shape
            .iter()
            .map(|&n| i64::try_from(n))
            .collect::<std::result::Result<_, _>>()?;
        let mut handle = ptr::null_mut();
        let mut error = [0i8; 4096];
        let status = unsafe {
            (self.create)(
                data.as_ptr(),
                data.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        ensure!(!handle.is_null(), "ATen returned null tensor");
        Ok(Tensor {
            handle,
            api: self.clone(),
        })
    }
    fn bytes(self: &Rc<Self>, data: &[u8], rows: usize, cols: usize) -> Result<Tensor> {
        ensure!(
            rows.checked_mul(cols) == Some(data.len())
                && !data.is_empty()
                && data.len() <= crate::weights::MAX_READ,
            "packed upload shape"
        );
        let mut handle = ptr::null_mut();
        let mut error = [0i8; 4096];
        check(
            unsafe {
                (self.bytes)(
                    data.as_ptr(),
                    rows as i64,
                    cols as i64,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        ensure!(!handle.is_null(), "null packed tensor");
        Ok(Tensor {
            handle,
            api: self.clone(),
        })
    }
    fn run(self: &Rc<Self>, op: i32, inputs: &[&Tensor], params: &[f64]) -> Result<Tensor> {
        ensure!(
            inputs.iter().all(|t| Rc::ptr_eq(&t.api, self)),
            "tensor belongs to another ATen runtime"
        );
        let handles: Vec<_> = inputs.iter().map(|t| t.handle).collect();
        let mut handle = ptr::null_mut();
        let mut error = [0i8; 4096];
        let status = unsafe {
            (self.run)(
                op,
                handles.as_ptr(),
                handles.len() as i64,
                params.as_ptr(),
                params.len() as i64,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        ensure!(!handle.is_null(), "ATen returned null result");
        Ok(Tensor {
            handle,
            api: self.clone(),
        })
    }
    pub fn vectors(
        self: &Rc<Self>,
        op: i32,
        inputs: &[&[f32]],
        params: &[f64],
        output: usize,
    ) -> Result<Vec<f32>> {
        let tensors: Vec<_> = inputs
            .iter()
            .map(|v| self.tensor(v, &[v.len()]))
            .collect::<Result<_>>()?;
        self.run(op, &tensors.iter().collect::<Vec<_>>(), params)?
            .read(output)
    }
    pub fn pre(self: &Rc<Self>, input: &[f32], pre: &[f32]) -> Result<Vec<f32>> {
        self.vectors(3, &[input, pre], &[], input.len() / 4)
    }
    pub fn post(self: &Rc<Self>, input: &[f32], residual: &[f32], mix: &Mix) -> Result<Vec<f32>> {
        self.vectors(
            4,
            &[input, residual, &mix.post, &mix.comb],
            &[],
            residual.len(),
        )
    }
}
fn check(status: i32, error: &[c_char]) -> Result<()> {
    if status != 0 {
        // The shim always terminates its bounded caller-owned error buffer.
        bail!(
            "native ATen: {}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
    }
    Ok(())
}
pub struct Tensor {
    handle: Handle,
    api: Rc<Api>,
}
impl Tensor {
    pub fn read(&self, count: usize) -> Result<Vec<f32>> {
        ensure!(
            count <= 1024 * 1024 * 1024 / 4,
            "readback exceeds 1 GiB cap"
        );
        let mut result = vec![0.; count];
        let mut error = [0i8; 4096];
        let status = unsafe {
            (self.api.copy)(
                self.handle,
                result.as_mut_ptr(),
                count as i64,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        check(status, &error)?;
        Ok(result)
    }
}
impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            (self.api.destroy)(self.handle);
        }
    }
}

pub struct Compatibility {
    pub api: Rc<Api>,
    weights: HashMap<String, Tensor>,
    rotations: Option<[Tensor; 2]>,
    pub resident_bytes: usize,
}

struct CachedProjection {
    weight: Tensor,
    scales: Option<Tensor>,
    bytes: usize,
    used: u64,
}
/// Quantized weights/scales remain packed on HIP; dequantized slabs are transient.
/// Dense converted weights are counted at their actual FP32 storage size. Host
/// reads remain bounded; no model mmap or whole Engram table is ever created.
pub struct Projector {
    pub api: Rc<Api>,
    cache: HashMap<String, CachedProjection>,
    clock: u64,
}
impl Projector {
    pub fn clear(&mut self) {
        self.cache.clear();
    }
    pub fn new(api: Rc<Api>) -> Self {
        Self {
            api,
            cache: HashMap::new(),
            clock: 0,
        }
    }
    fn reserve(&mut self, stats: &mut crate::weights::Stats, size: usize) -> Result<()> {
        ensure!(
            size <= stats.budget_bytes,
            "ATen weight exceeds cache budget"
        );
        while stats.resident_bytes + size > stats.budget_bytes {
            let victim = self
                .cache
                .iter()
                .min_by_key(|(_, v)| v.used)
                .map(|(k, _)| k.clone())
                .context("ATen cache eviction")?;
            let old = self.cache.remove(&victim).unwrap();
            stats.resident_bytes -= old.bytes;
        }
        Ok(())
    }
    fn head(
        &mut self,
        store: &mut Store,
        stats: &mut crate::weights::Stats,
        x: &Tensor,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let key = "head.weight/full";
        self.clock += 1;
        if !self.cache.contains_key(key) {
            let started = std::time::Instant::now();
            let size = rows
                .checked_mul(cols)
                .and_then(|n| n.checked_mul(4))
                .context("head size overflow")?;
            self.reserve(stats, size)?;
            store.expect("head.weight", &[rows, cols], safetensors::Dtype::BF16)?;
            let weight = self.api.run(11, &[x], &[rows as f64, cols as f64])?;
            for start in (0..rows).step_by(2048) {
                let count = 2048.min(rows - start);
                let bytes = store.bytes("head.weight", start * cols * 2, count * cols * 2)?;
                let data: Vec<f32> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| half::bf16::from_le_bytes(*v).to_f32())
                    .collect();
                stats.peak_host_weight_staging_bound_bytes = stats
                    .peak_host_weight_staging_bound_bytes
                    .max(bytes.len() + data.len() * 4);
                let chunk = self.api.tensor(&data, &[count, cols])?;
                self.api.run(12, &[&weight, &chunk], &[start as f64])?;
            }
            self.cache.insert(
                key.into(),
                CachedProjection {
                    weight,
                    scales: None,
                    bytes: size,
                    used: self.clock,
                },
            );
            stats.resident_bytes += size;
            stats.peak_resident_bytes = stats.peak_resident_bytes.max(stats.resident_bytes);
            stats.uploads += 1;
            stats.upload_seconds += started.elapsed().as_secs_f64();
        } else {
            stats.cache_hits += 1;
        }
        let weight = self.cache.get_mut(key).unwrap();
        weight.used = self.clock;
        let started = std::time::Instant::now();
        let result = self.api.run(5, &[x, &weight.weight], &[2.])?.read(rows)?;
        stats.matvecs += 1;
        stats.matvec_seconds += started.elapsed().as_secs_f64();
        Ok(result)
    }
    #[allow(clippy::too_many_arguments)] // Same explicit checked range contract as Vulkan.
    pub fn range(
        &mut self,
        store: &mut Store,
        stats: &mut crate::weights::Stats,
        name: &str,
        input: &[f32],
        first: usize,
        rows: usize,
        round: bool,
    ) -> Result<Vec<f32>> {
        use crate::math;
        use safetensors::Dtype;
        let info = store.info(name)?.clone();
        ensure!(
            info.shape.len() == 2
                && rows > 0
                && first.checked_add(rows).is_some_and(|n| n <= info.shape[0]),
            "matrix range"
        );
        let fp4 = info.dtype == Dtype::I8 && name.contains(".ffn.experts.");
        let quant = fp4 || info.dtype == Dtype::F8_E4M3;
        let cols = info.shape[1] * if fp4 { 2 } else { 1 };
        ensure!(cols == input.len(), "native ATen matrix input shape");
        let dense_group = name.ends_with(".wo_a.weight");
        let mut x = input.to_vec();
        if quant && !dense_group {
            math::quant_fp8(&mut x)?;
        }
        let x = self.api.tensor(&x, &[1, cols])?;
        if name == "head.weight" {
            ensure!(
                first == 0 && rows == info.shape[0] && !round,
                "head must use full original matrix shape"
            );
            return self.head(store, stats, &x, rows, cols);
        }
        // Quantized reference GEMMs dispatch 256-row full-K slabs. Dense ops use
        // their original shape; the head is assembled from bounded reads above.
        let block = if quant && !dense_group {
            256.min((16 * 1024 * 1024 / (4 * cols)).max(1))
        } else {
            rows.min((crate::weights::MAX_READ / (4 * cols)).max(1))
        };
        let mut result = Vec::with_capacity(rows);
        for start in (first..first + rows).step_by(block) {
            let count = block.min(first + rows - start);
            let key = format!("{name}/{start}/{count}");
            self.clock += 1;
            if !self.cache.contains_key(&key) {
                let time = std::time::Instant::now();
                let size = if quant {
                    count * info.shape[1] + count.div_ceil(if fp4 { 1 } else { 32 }) * (cols / 32)
                } else {
                    count * cols * 4
                };
                self.reserve(stats, size)?;
                let width = match info.dtype {
                    Dtype::F32 => 4,
                    Dtype::BF16 => 2,
                    Dtype::F8_E4M3 => 1,
                    Dtype::I8 if fp4 => 1,
                    _ => bail!("unsupported ATen weight dtype"),
                };
                let bytes = store.bytes(
                    name,
                    start * info.shape[1] * width,
                    count * info.shape[1] * width,
                )?;
                let mut scales = vec![];
                if quant {
                    ensure!(
                        start.is_multiple_of(32) && cols.is_multiple_of(32),
                        "quantized slab alignment"
                    );
                    let divisor = if fp4 { 1 } else { 32 };
                    let scale_name = name
                        .strip_suffix("weight")
                        .context("scale name")?
                        .to_owned()
                        + "scale";
                    store.expect(
                        &scale_name,
                        &[info.shape[0].div_ceil(divisor), cols / 32],
                        Dtype::F8_E8M0,
                    )?;
                    scales = store.bytes(
                        &scale_name,
                        start / divisor * (cols / 32),
                        count.div_ceil(divisor) * (cols / 32),
                    )?;
                }
                let (weight, scale_tensor) = if quant {
                    ensure!(!scales.contains(&255), "nonfinite quantized scale");
                    (
                        self.api.bytes(&bytes, count, info.shape[1])?,
                        Some(self.api.bytes(
                            &scales,
                            count.div_ceil(if fp4 { 1 } else { 32 }),
                            cols / 32,
                        )?),
                    )
                } else {
                    let data: Vec<f32> = if info.dtype == Dtype::BF16 {
                        bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| half::bf16::from_le_bytes(*v).to_f32())
                            .collect()
                    } else {
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f32::from_le_bytes(*v))
                            .collect()
                    };
                    (self.api.tensor(&data, &[count, cols])?, None)
                };
                stats.peak_host_weight_staging_bound_bytes = stats
                    .peak_host_weight_staging_bound_bytes
                    .max(bytes.len() + scales.len() + if quant { 0 } else { size });
                self.cache.insert(
                    key.clone(),
                    CachedProjection {
                        weight,
                        scales: scale_tensor,
                        bytes: size,
                        used: self.clock,
                    },
                );
                stats.resident_bytes += size;
                stats.peak_resident_bytes = stats.peak_resident_bytes.max(stats.resident_bytes);
                stats.uploads += 1;
                stats.upload_seconds += time.elapsed().as_secs_f64();
            } else {
                stats.cache_hits += 1;
            }
            let w = self.cache.get_mut(&key).unwrap();
            w.used = self.clock;
            let mode = if quant && !dense_group {
                0.
            } else if round {
                1.
            } else {
                2.
            };
            let time = std::time::Instant::now();
            let output = if let Some(scales) = &w.scales {
                self.api.run(
                    13,
                    &[&x, &w.weight, scales],
                    &[if fp4 { 4. } else { 8. }, f64::from(dense_group)],
                )?
            } else {
                self.api.run(5, &[&x, &w.weight], &[mode])?
            };
            result.extend(output.read(count)?);
            stats.matvecs += 1;
            stats.matvec_seconds += time.elapsed().as_secs_f64();
        }
        Ok(result)
    }
}
impl Compatibility {
    pub fn clear(&mut self) {
        self.weights.clear();
        self.rotations = None;
        self.resident_bytes = 0;
    }
    pub fn new(api: Rc<Api>) -> Self {
        Self {
            api,
            weights: HashMap::new(),
            rotations: None,
            resident_bytes: 0,
        }
    }
    pub fn init_rotary(&mut self, cfg: &crate::config::Config, context: usize) -> Result<()> {
        let dummy = self.api.tensor(&[0.], &[1])?;
        let make = |compressed: bool| {
            self.api.run(
                15,
                &[&dummy],
                &[
                    cfg.rope_head_dim as f64,
                    context as f64,
                    if compressed {
                        cfg.original_seq_len as f64
                    } else {
                        0.
                    },
                    if compressed {
                        cfg.compress_rope_theta as f64
                    } else {
                        cfg.rope_theta as f64
                    },
                    cfg.rope_factor as f64,
                    cfg.beta_fast as f64,
                    cfg.beta_slow as f64,
                ],
            )
        };
        self.rotations = Some([make(false)?, make(true)?]);
        Ok(())
    }
    pub fn rotary(
        &self,
        x: &mut [f32],
        compressed: bool,
        position: usize,
        head_dim: usize,
        rope_dim: usize,
        inverse: bool,
    ) -> Result<()> {
        let tables = self
            .rotations
            .as_ref()
            .context("ATen rotary tables not initialized")?;
        let input = self.api.tensor(x, &[x.len()])?;
        let result = self
            .api
            .run(
                16,
                &[&input, &tables[usize::from(compressed)]],
                &[
                    head_dim as f64,
                    rope_dim as f64,
                    position as f64,
                    f64::from(inverse),
                ],
            )?
            .read(x.len())?;
        x.copy_from_slice(&result);
        Ok(())
    }
    pub fn mix(
        &mut self,
        store: &mut Store,
        prefix: &str,
        x: &[f32],
        eps: f32,
        norm_eps: f32,
        iters: usize,
    ) -> Result<Vec<f32>> {
        let name = format!("{prefix}_fn");
        let info = store.info(&name)?.clone();
        ensure!(
            info.dtype == safetensors::Dtype::F32 && info.shape == [24, x.len()],
            "HC checkpoint shape"
        );
        if !self.weights.contains_key(&name) {
            ensure!(
                self.resident_bytes + 24 * x.len() * 4 <= 256 * 1024 * 1024,
                "HC resident cache exceeds 256 MiB"
            );
            let bytes = store.bytes(&name, 0, 24 * x.len() * 4)?;
            let data: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect();
            let weight = self.api.tensor(&data, &[24, x.len()])?;
            self.resident_bytes += bytes.len();
            self.weights.insert(name.clone(), weight);
        }
        let input = self.api.tensor(x, &[1, 1, x.len()])?;
        let scale = self
            .api
            .tensor(&store.small(&format!("{prefix}_scale"))?, &[3])?;
        let base = self
            .api
            .tensor(&store.small(&format!("{prefix}_base"))?, &[24])?;
        self.api
            .run(
                1,
                &[&input, &self.weights[&name], &scale, &base],
                &[eps as f64, norm_eps as f64, iters as f64],
            )?
            .read(73)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_runtime_fails_without_fallback() {
        assert!(
            Api::open(
                Path::new("Z:/missing-strix-sdk"),
                Path::new("Z:/missing-strix-shim.dll")
            )
            .is_err()
        );
    }
    #[test]
    fn rejects_python_and_preloaded_dll_conflicts() -> Result<()> {
        assert!(reject_python(&[PathBuf::from("C:/runtime/python312.dll")]).is_err());
        assert!(reject_python(&[PathBuf::from("C:/runtime/torch_python.dll")]).is_err());
        let directory =
            std::env::temp_dir().join(format!("strix-dll-conflict-{}", std::process::id()));
        std::fs::create_dir(&directory)?;
        let fake = directory.join("kernel32.dll");
        std::fs::write(&fake, b"not a DLL")?;
        let result = validate_module(&fake);
        std::fs::remove_file(&fake)?;
        std::fs::remove_dir(&directory)?;
        assert!(result.unwrap_err().to_string().contains("DLL conflict"));
        Ok(())
    }
    #[test]
    #[ignore = "requires explicitly configured native ATen/HIP shim and Radeon 8060S"]
    fn hip_ffi_lifecycle_and_errors() -> Result<()> {
        let api = Api::from_env()?;
        assert!(api.tensor(&[1.], &[2]).is_err());
        assert!(api.tensor(&[f32::NAN], &[1]).is_err());
        let mut steady_allocated = None;
        for _ in 0..16 {
            {
                let x = api.tensor(&[1.; 4], &[1, 1, 4])?;
                let w = api.tensor(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[3, 4])?;
                assert_eq!(api.run(0, &[&x, &w], &[])?.read(3)?, [6., 22., 38.]);
                assert!(api.run(0, &[&x, &x], &[]).is_err());
                assert!(api.run(999, &[&x], &[]).is_err());
                for bad in [f64::MAX, f64::NAN, -1., 1.5] {
                    assert!(api.run(11, &[&x], &[bad, 1.]).is_err());
                }
                assert!(x.read(3).is_err());
            }
            let allocated = api.memory()?["allocated_bytes"]
                .as_i64()
                .context("allocator counter")?;
            if let Some(expected) = steady_allocated {
                assert_eq!(
                    allocated, expected,
                    "active tensor allocation leaked across FFI errors"
                );
            } else {
                steady_allocated = Some(allocated);
            }
        }
        // Original France prefix, position 5, layer 23 query components 1162/3.
        // The scalar complex product crosses a BF16 rounding boundary here.
        // Keep the original 64-head/64-rotary-component dispatch shape. A
        // two-element tensor uses another ATen kernel and is not this oracle.
        let mut query = vec![0.; 4096];
        query[1162..1164].copy_from_slice(&[1.0625, -1.1953125]);
        let mut cos = [1.; 32];
        let mut sin = [0.; 32];
        cos[5] = 0.718_729_7;
        sin[5] = 0.695_289_6;
        let rotated = api.vectors(17, &[&query, &cos, &sin], &[64., 64., 0., 0.], 4096)?;
        assert_eq!(&rotated[1162..1164], &[1.59375, -0.12060547]);
        // Raw ABI failures leave the caller's handle null and an error string;
        // subsequent valid calls remain usable. No C++ exception crosses FFI.
        let mut handle = ptr::null_mut();
        let mut error = [0i8; 4096];
        let shape = [1i64];
        let status = unsafe {
            (api.create)(
                ptr::null(),
                1,
                shape.as_ptr(),
                1,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_ne!(status, 0);
        assert!(handle.is_null());
        assert_ne!(error[0], 0);
        reject_python(&loaded_modules()?)?;
        Ok(())
    }
}
