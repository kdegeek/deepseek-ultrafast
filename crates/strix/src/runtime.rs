use crate::reference::{self, Format};
use anyhow::{Context, Result, ensure};
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt;

#[derive(Debug, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub driver: String,
    pub backend: String,
    /// A per-binding limit, NOT available GPU or shared system memory.
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    format: u32,
    epsilon: f32,
}

/// Explicit hardware-only Vulkan selection. Never falls back to a software adapter.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    matvec: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
    pub info: DeviceInfo,
}

pub struct Matrix {
    weights: wgpu::Buffer,
    scales: wgpu::Buffer,
    rows: usize,
    cols: usize,
    format: Format,
    // Prevent mixing buffers belonging to different devices.
    device: wgpu::Device,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter().find(|a| {
                let i = a.get_info();
                i.vendor == 0x1002 && matches!(i.device_type, wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu)
            }).context("No AMD hardware Vulkan adapter found; install an AMD Vulkan-capable graphics driver")?;
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        ensure!(
            limits.max_compute_invocations_per_workgroup >= 256
                && limits.max_compute_workgroup_size_x >= 256,
            "AMD adapter must support 256-lane compute workgroups"
        );
        let info = DeviceInfo {
            name: adapter_info.name,
            vendor: adapter_info.vendor,
            device: adapter_info.device,
            driver: adapter_info.driver_info,
            backend: format!("{:?}", adapter_info.backend),
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("Strix Halo compute"),
                required_limits: limits,
                ..Default::default()
            }))?;
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strix primitives"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ops.wgsl").into()),
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let matvec = pipeline("matvec");
        let rms_norm = pipeline("rms_norm");
        let silu_mul = pipeline("silu_mul");
        Ok(Self {
            device,
            queue,
            layout,
            matvec,
            rms_norm,
            silu_mul,
            info,
        })
    }

    fn storage(&self, label: &str, bytes: &[u8]) -> Result<wgpu::Buffer> {
        let bytes = if bytes.is_empty() { &[0u8; 4] } else { bytes };
        self.check_size(bytes.len())?;
        ensure!(
            bytes.len().is_multiple_of(4),
            "buffer must be padded to a whole u32"
        );
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }))
    }

    fn check_size(&self, bytes: usize) -> Result<()> {
        ensure!(
            bytes as u64 <= self.info.max_buffer_size
                && bytes as u64 <= self.info.max_storage_buffer_binding_size,
            "buffer of {bytes} bytes exceeds device storage binding limit {}",
            self.info.max_storage_buffer_binding_size
        );
        Ok(())
    }

    pub fn upload_matrix(
        &self,
        weights: &[u32],
        scales: &[u32],
        rows: usize,
        cols: usize,
        format: Format,
    ) -> Result<Matrix> {
        reference::validate_matrix(weights, scales, rows, cols, format)?;
        self.dimensions(rows, cols)?;
        Ok(Matrix {
            weights: self.storage("weights", bytemuck::cast_slice(weights))?,
            scales: self.storage("scales", bytemuck::cast_slice(scales))?,
            rows,
            cols,
            format,
            device: self.device.clone(),
        })
    }

    fn dimensions(&self, rows: usize, cols: usize) -> Result<()> {
        ensure!(rows > 0 && cols > 0, "empty dimensions");
        ensure!(
            rows <= self.device.limits().max_compute_workgroups_per_dimension as usize,
            "too many output rows for one dispatch"
        );
        ensure!(
            rows.checked_mul(cols)
                .is_some_and(|n| n <= u32::MAX as usize),
            "matrix indexing exceeds u32"
        );
        Ok(())
    }

    pub fn matvec(&self, matrix: &Matrix, input: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            matrix.device == self.device,
            "matrix belongs to another device"
        );
        ensure!(
            input.len() == matrix.cols,
            "input must match matrix columns"
        );
        ensure!(input.iter().all(|x| x.is_finite()), "input must be finite");
        let input = self.storage("input", bytemuck::cast_slice(input))?;
        let auxiliary = self.storage("unused", &[])?;
        self.execute(
            &self.matvec,
            &matrix.weights,
            &matrix.scales,
            &input,
            &auxiliary,
            Params {
                rows: matrix.rows as u32,
                cols: matrix.cols as u32,
                format: matrix.format as u32,
                epsilon: 0.0,
            },
            matrix.rows,
            matrix.rows as u32,
        )
    }

    pub fn rms_norm(
        &self,
        input: &[f32],
        gamma: &[f32],
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>> {
        self.dimensions(rows, cols)?;
        ensure!(
            input.len() == rows * cols && gamma.len() == cols,
            "RMSNorm shape mismatch"
        );
        ensure!(
            input.iter().chain(gamma).all(|x| x.is_finite()),
            "RMSNorm inputs must be finite"
        );
        ensure!(
            epsilon.is_finite() && epsilon > 0.0,
            "epsilon must be finite and positive"
        );
        let input_buf = self.storage("input", bytemuck::cast_slice(input))?;
        let gamma = self.storage("gamma", bytemuck::cast_slice(gamma))?;
        let unused = self.storage("unused", &[])?;
        self.execute(
            &self.rms_norm,
            &unused,
            &unused,
            &input_buf,
            &gamma,
            Params {
                rows: rows as u32,
                cols: cols as u32,
                format: 0,
                epsilon,
            },
            input.len(),
            rows as u32,
        )
    }

    pub fn silu_mul(&self, input: &[f32], other: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            !input.is_empty() && input.len() == other.len(),
            "SiLU shape mismatch or empty input"
        );
        ensure!(
            input.iter().chain(other).all(|x| x.is_finite()),
            "SiLU inputs must be finite"
        );
        self.dimensions(input.len().div_ceil(64), 64)?;
        let a = self.storage("gate", bytemuck::cast_slice(input))?;
        let b = self.storage("up", bytemuck::cast_slice(other))?;
        let unused = self.storage("unused", &[])?;
        self.execute(
            &self.silu_mul,
            &unused,
            &unused,
            &a,
            &b,
            Params {
                rows: 1,
                cols: input.len() as u32,
                format: 0,
                epsilon: 0.0,
            },
            input.len(),
            input.len().div_ceil(64) as u32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        weights: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        input: &wgpu::Buffer,
        auxiliary: &wgpu::Buffer,
        params: Params,
        elements: usize,
        groups: u32,
    ) -> Result<Vec<f32>> {
        let size = elements.checked_mul(4).context("output size overflow")?;
        self.check_size(size)?;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let entries: Vec<_> = [weights, scales, input, auxiliary, &output, &uniform]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size as u64);
        let submission = self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })?;
        rx.recv_timeout(Duration::from_secs(5))
            .context("GPU readback callback timed out")??;
        let mapped = readback.slice(..).get_mapped_range()?;
        let result = mapped
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(result)
    }
}
