"""Bounded lazy loading of the original DeepSeek safetensors checkpoint.

Parameters live on meta until their owning module is called. A size-bounded LRU
keeps device tensors; active forward scopes are pinned. Engram embeddings fetch
only requested rows with bounded file reads. This is an inference-only, single-
device, single-threaded reference loader, not a concurrent serving backend.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import json
import operator
import os
from pathlib import Path
import struct
import sys
import threading
import time
import types

import torch
from safetensors import safe_open


MAX_ROW_HEADER_BYTES = 32 * 1024**2
MAX_ROW_READ_BYTES = 1024**2
_ROW_DTYPES = {name: getattr(torch, dtype) for name, dtype in {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64",
    "U16": "uint16", "U32": "uint32", "U64": "uint64", "F16": "float16", "BF16": "bfloat16",
    "F32": "float32", "F64": "float64", "C64": "complex64",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu",
    "F8_E4M3FNUZ": "float8_e4m3fnuz", "F8_E5M2FNUZ": "float8_e5m2fnuz",
}.items()}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _RowMetadata:
    shape: tuple
    dtype: str
    offset: int
    size: int

    def get_shape(self):
        return list(self.shape)

    def get_dtype(self):
        return self.dtype


class CheckpointStore:
    """Read dense tensors individually and sparse rows without PyTorch mmap storage.

    Checkpoint files must remain immutable while the store is open. Sparse reads
    are single-threaded, like the owning model, and preserve requested row order.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._stack = ExitStack()
        self._files = {}
        self._row_files = {}
        self._row_metadata = {}
        self.row_read_bytes = 0
        self.peak_row_read_bytes = 0
        self.closed = False
        try:
            self._read_index()
        except BaseException:
            self.close()
            raise

    def _read_index(self):
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            with index.open(encoding="utf-8") as stream:
                self.weight_map = json.load(stream)["weight_map"]
            if not isinstance(self.weight_map, dict) or not self.weight_map:
                raise ValueError("checkpoint index must have a nonempty weight_map")
            for name, filename in self.weight_map.items():
                if not isinstance(name, str) or not isinstance(filename, str):
                    raise ValueError("checkpoint index names must be strings")
                path = Path(filename)
                if path.name != filename or any(c in filename for c in ("/", "\\", ":")):
                    raise ValueError(f"invalid checkpoint shard path: {filename}")
                if not (self.directory / filename).is_file():
                    raise FileNotFoundError(f"missing checkpoint shard: {filename}")
        else:
            self.weight_map = {}
            for path in sorted(self.directory.glob("*.safetensors")):
                handle = self._open(path.name)
                for name in handle.keys():
                    if name in self.weight_map:
                        raise ValueError(f"duplicate checkpoint tensor: {name}")
                    self.weight_map[name] = path.name
            if not self.weight_map:
                raise ValueError(f"no safetensors tensors in {self.directory}")

    def _open(self, filename, backend="pread"):
        if self.closed:
            raise RuntimeError("checkpoint store is closed")
        key = (filename, backend)
        if key not in self._files:
            with torch.device("cpu"):
                self._files[key] = self._stack.enter_context(
                    safe_open(self.directory / filename, framework="pt", device="cpu", backend=backend)
                )
        return self._files[key]

    def resolve(self, name):
        """Accept the original names and the official converter's HF aliases."""
        candidates = [name, "model." + name]
        hf_name = name.replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
        if hf_name.endswith(".scale"):
            hf_name = hf_name[:-6] + ".weight_scale_inv"
        if hf_name.endswith(".gate.bias"):
            hf_name = hf_name[:-5] + ".e_score_correction_bias"
        candidates += [hf_name, "model." + hf_name]
        for candidate in candidates:
            if candidate in self.weight_map:
                return candidate
        raise KeyError(f"checkpoint tensor not found: {name}")

    def slice(self, name):
        """Metadata only; payload slicing is intentionally not exposed."""
        name = self.resolve(name)
        if self.closed:
            raise RuntimeError("checkpoint store is closed")
        if name not in self._row_metadata:
            filename = self.weight_map[name]
            if filename not in self._row_files:
                stream = (self.directory / filename).open("rb", buffering=0)
                try:
                    prefix = stream.read(8)
                    if len(prefix) != 8:
                        raise ValueError("truncated safetensors header length")
                    length = struct.unpack("<Q", prefix)[0]
                    file_size = os.fstat(stream.fileno()).st_size
                    if not 0 < length <= MAX_ROW_HEADER_BYTES or 8 + length > file_size:
                        raise ValueError("invalid or oversized safetensors row header")
                    header_bytes = stream.read(length)
                    if len(header_bytes) != length:
                        raise EOFError("truncated safetensors row header")
                    header = json.loads(header_bytes, object_pairs_hook=_unique_object)
                    # The library validates all shapes, contiguous offsets, dtype
                    # widths and complete payload length. get_slice below only
                    # retrieves metadata; __getitem__ is never called because
                    # safetensors 0.8 pread slicing reads the entire tensor.
                    self._open(filename)
                    self._stack.enter_context(stream)
                    self._row_files[filename] = (stream, header, 8 + length, file_size)
                except BaseException:
                    stream.close()
                    raise
            _, header, data_start, file_size = self._row_files[filename]
            info = header[name]
            view = self._open(filename).get_slice(name)
            shape, dtype = view.get_shape(), view.get_dtype()
            begin, end = info["data_offsets"]
            if (info["shape"] != shape or info["dtype"] != dtype
                    or type(begin) is not int or type(end) is not int
                    or not 0 <= begin <= end <= file_size - data_start):
                raise ValueError(f"{name}: inconsistent safetensors row metadata")
            self._row_metadata[name] = _RowMetadata(tuple(shape), dtype, data_start + begin, end - begin)
        return self._row_metadata[name]

    def tensor(self, name):
        name = self.resolve(name)
        # The many-shard PyTorch mmap-storage path access-violates on this Windows
        # AMD runtime. Pread owns just this tensor's CPU bytes and avoids that path.
        # Keep its factory calls on CPU even inside a default CUDA device context.
        with torch.device("cpu"):
            return self._open(self.weight_map[name]).get_tensor(name)

    def rows(self, name, ids):
        """Read selected rows only, including on platforms without FP8 indexing."""
        name = self.resolve(name)
        view = self.slice(name)
        shape = view.get_shape()
        if len(shape) != 2:
            raise ValueError(f"{name}: expected two-dimensional row table")
        if not isinstance(ids, (list, tuple, range)) or not ids:
            raise ValueError("row request must be a nonempty list, tuple or range of integers")
        if view.dtype not in _ROW_DTYPES:
            raise ValueError(f"{name}: unsupported byte-aligned row dtype {view.dtype}")
        dtype = _ROW_DTYPES[view.dtype]
        stride = shape[1] * dtype.itemsize
        size = len(ids) * stride
        if not stride or size > MAX_ROW_READ_BYTES:
            raise ValueError(f"row request must contain 1..{MAX_ROW_READ_BYTES} bytes")
        if view.size != shape[0] * stride:
            raise ValueError(f"{name}: row shape/dtype does not match tensor byte length")
        if any(isinstance(i, bool) for i in ids):
            raise TypeError("row IDs must be integers, not booleans")
        ids = [operator.index(i) for i in ids]
        if any(i < 0 or i >= shape[0] for i in ids):
            raise IndexError(f"{name}: row outside checkpoint table")
        stream = self._row_files[self.weight_map[name]][0]
        data = bytearray(size)
        target = memoryview(data)
        for out, row in enumerate(ids):
            stream.seek(view.offset + row * stride)
            begin, end = out * stride, (out + 1) * stride
            while begin < end:
                count = stream.readinto(target[begin:end])
                if not count:
                    raise EOFError(f"{name}: checkpoint row truncated")
                begin += count
        self.row_read_bytes += size
        self.peak_row_read_bytes = max(self.peak_row_read_bytes, size)
        # Own one bounded byte buffer. frombuffer retains its Python owner; no
        # mapped storage, dtype conversion, shell-FP8 indexing or extra copy.
        with torch.device("cpu"):
            return torch.frombuffer(data, dtype=torch.uint8).view(dtype).reshape(len(ids), shape[1])

    def close(self):
        self.closed = True
        self._files.clear()
        self._row_files.clear()
        self._row_metadata.clear()
        self._stack.close()


@dataclass
class _Parameter:
    module: torch.nn.Module
    attribute: str
    shape: tuple
    dtype: torch.dtype
    meta: torch.nn.Parameter
    size: int


class LazyCheckpoint:
    def __init__(self, model, checkpoint_dir, device, cache_bytes, retain_bytes=0):
        if not isinstance(cache_bytes, int) or cache_bytes <= 0:
            raise ValueError("cache_bytes must be a positive integer")
        self.model = model
        self.device = torch.device(device)
        self.cache_bytes = cache_bytes
        if not isinstance(retain_bytes, int) or not 0 <= retain_bytes <= cache_bytes // 2:
            raise ValueError("retain_bytes must be between zero and half the weight cache")
        self.retain_bytes = retain_bytes
        self.retained = set()
        self.store = CheckpointStore(checkpoint_dir)
        self.cache = OrderedDict()
        self.specs = {}
        self.pins = {}
        self.handles = []
        self._patched_forwards = []
        self._scope_stacks = {}
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.loads = 0
        self.cache_hits = 0
        self.engram_rows = 0
        self.loaded_bytes = 0
        self.hit_bytes = 0
        self.load_seconds = 0.0
        self.closed = False
        try:
            self._attach()
        except BaseException:
            self.close()
            raise

    def _attach(self):
        modules = dict(self.model.named_modules())
        sparse = {name for name, module in modules.items()
                  if module.__class__.__name__ == "ParallelEngramEmbedding"}
        scopes = {}
        for prefix, module in modules.items():
            if prefix in sparse:
                self._attach_engram(prefix, module)
                continue
            keys = []
            for attribute, parameter in module.named_parameters(recurse=False):
                name = f"{prefix}.{attribute}" if prefix else attribute
                self.store.resolve(name)  # missing parameters fail before generation
                meta = torch.nn.Parameter(parameter.detach().to("meta"), requires_grad=False)
                self.specs[name] = _Parameter(module, attribute, tuple(parameter.shape),
                                             parameter.dtype, meta,
                                             parameter.numel() * parameter.element_size())
                module._parameters[attribute] = meta
                keys.append(name)
            scopes[module] = keys
        # Reserve a bounded subset of frequently reused non-routed parameters.
        # Smallest first avoids a large head displacing hundreds of shared weights.
        remaining = self.retain_bytes
        for name, spec in sorted(self.specs.items(), key=lambda item: (item[1].size, item[0])):
            if ".experts." not in name and spec.size <= remaining:
                self.retained.add(name)
                remaining -= spec.size
        # The official Attention path accesses wo_a.weight directly for einsum,
        # so wo_a's own __call__ hooks are not invoked.
        for prefix, module in modules.items():
            wo_a = getattr(module, "wo_a", None)
            if wo_a is not None:
                scopes.setdefault(module, []).extend(scopes.get(wo_a, []))
        for module, keys in scopes.items():
            if keys:
                self._scope_stacks[module] = []
                self.handles.append(module.register_forward_pre_hook(self._pre(tuple(keys))))
                self.handles.append(module.register_forward_hook(self._post, always_call=True))

    def _pre(self, keys):
        def enter(module, inputs):
            if self.closed:
                raise RuntimeError("lazy checkpoint is closed")
            acquired = []
            self._scope_stacks[module].append(acquired)
            for name in keys:
                self._acquire(name)
                acquired.append(name)
            # Tensor attributes do not survive parameter replacement. Reattach
            # scale to weight after both have been loaded for this scope.
            for name in keys:
                owner = self.specs[name].module
                weight, scale = getattr(owner, "weight", None), getattr(owner, "scale", None)
                if weight is not None and scale is not None:
                    weight.scale = scale
        return enter

    def _post(self, module, inputs, output):
        stack = self._scope_stacks[module]
        if stack:
            for name in stack.pop():
                self.pins[name] -= 1

    def _evict(self, required):
        if required > self.cache_bytes:
            raise MemoryError(f"parameter needs {required} bytes, above {self.cache_bytes}-byte weight cache")
        while self.resident_bytes + required > self.cache_bytes:
            victim = next((key for key in self.cache
                           if key not in self.retained and self.pins.get(key, 0) == 0), None)
            if victim is None:
                raise MemoryError("active or retained parameters exceed the weight cache budget; lower retention or increase cache")
            spec = self.specs[victim]
            spec.module._parameters[spec.attribute] = spec.meta
            # Drop the extra reference carried by quantized weight.scale.
            if spec.attribute == "scale":
                weight = getattr(spec.module, "weight", None)
                if weight is not None and hasattr(weight, "scale"):
                    weight.scale = spec.meta
            self.resident_bytes -= spec.size
            del self.cache[victim]

    def _materialize(self, name, spec):
        source = self.store.tensor(name)
        if tuple(source.shape) != spec.shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(source.shape)} != model {spec.shape}")
        if spec.dtype == torch.float4_e2m1fn_x2:
            if source.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2):
                raise TypeError(f"{name}: expected packed FP4 bytes, got {source.dtype}")
            source = source.view(torch.float4_e2m1fn_x2)
        elif source.dtype == torch.float8_e4m3fn and spec.dtype in (torch.bfloat16, torch.float32):
            # Official convert.py dequantizes wo_a for the block-diagonal einsum.
            if not name.endswith(".wo_a.weight") or len(spec.shape) != 2:
                raise TypeError(f"{name}: unsupported FP8-to-dense conversion")
            scale = self.store.tensor(name[:-7] + ".scale")
            rows, cols = spec.shape
            if scale.ndim != 2 or rows % scale.shape[0] or cols % scale.shape[1]:
                raise ValueError(f"{name}: invalid dequantization scale shape")
            br, bc = rows // scale.shape[0], cols // scale.shape[1]
            if (br, bc) not in ((32, 32), (128, 128)):
                raise ValueError(f"{name}: unsupported scale block {br}x{bc}")
            source = (source.float().reshape(rows // br, br, cols // bc, bc)
                      * scale.float()[:, None, :, None]).reshape(rows, cols)
        elif source.dtype != spec.dtype and not (
            source.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and spec.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise TypeError(f"{name}: unsupported dtype conversion {source.dtype} -> {spec.dtype}")
        # copy=True is necessary even on CPU: cached tensors must own their bytes,
        # independent of the temporary CPU read buffer.
        return torch.nn.Parameter(source.to(device=self.device, dtype=spec.dtype, copy=True),
                                  requires_grad=False)

    def _acquire(self, name):
        spec = self.specs[name]
        if name in self.cache:
            self.cache_hits += 1
            self.hit_bytes += spec.size
            self.cache.move_to_end(name)
        else:
            self._evict(spec.size)
            started = time.perf_counter()
            value = self._materialize(name, spec)
            self.load_seconds += time.perf_counter() - started
            self.loaded_bytes += spec.size
            self.cache[name] = value
            self.resident_bytes += spec.size
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
            self.loads += 1
        self.pins[name] = self.pins.get(name, 0) + 1
        spec.module._parameters[spec.attribute] = self.cache[name]

    def _attach_engram(self, prefix, module):
        if module.vocab_start_idx != 0 or module.vocab_end_idx != module.num_embeddings:
            raise ValueError("lazy Engram loader supports only unsharded single-device models")
        weight_name, scale_name = prefix + ".weight", prefix + ".scale"
        if self.store.slice(weight_name).get_shape() != list(module.weight.shape):
            raise ValueError(f"{weight_name}: Engram shape mismatch")
        if self.store.slice(scale_name).get_shape() != list(module.scale.shape):
            raise ValueError(f"{scale_name}: Engram scale shape mismatch")
        module._parameters["weight"] = torch.nn.Parameter(module.weight.detach().to("meta"), requires_grad=False)
        module._parameters["scale"] = torch.nn.Parameter(module.scale.detach().to("meta"), requires_grad=False)
        original = module.forward
        self._patched_forwards.append((module, original))

        @torch.inference_mode()
        def forward(owner, indices):
            if self.closed:
                raise RuntimeError("lazy checkpoint is closed")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Engram indices must be integer tensors")
            flat = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            if flat.numel() == 0:
                return torch.empty((*indices.shape, owner.dim), device=indices.device, dtype=torch.bfloat16)
            mask = (flat < 0) | (flat >= owner.num_embeddings)
            valid = flat.masked_fill(mask, 0)
            unique, inverse = torch.unique(valid, sorted=True, return_inverse=True)
            ids = unique.tolist()
            values = self.store.rows(weight_name, ids)
            scales = self.store.rows(scale_name, ids)
            if values.dtype != torch.float8_e4m3fn or scales.dtype != torch.float8_e8m0fnu:
                raise TypeError("Engram requires original E4M3 weights and E8M0 scales")
            # Dequantize only unique fetched rows on CPU; this also works when the
            # GPU lacks shell-dtype indexing or E8M0 conversion kernels.
            result = (values.float().reshape(len(ids), -1, owner.block_size)
                      * scales.float().unsqueeze(-1)).reshape(len(ids), owner.dim)
            result = result.to(torch.bfloat16).index_select(0, inverse)
            result[mask] = 0
            self.engram_rows += len(ids)
            return result.reshape(*indices.shape, owner.dim).to(indices.device)

        module.forward = types.MethodType(forward, module)

    def stats(self):
        return {"cache_budget_bytes": self.cache_bytes, "resident_weight_bytes": self.resident_bytes,
                "peak_resident_weight_bytes": self.peak_resident_bytes, "parameter_loads": self.loads,
                "cache_hits": self.cache_hits, "engram_rows_read": self.engram_rows,
                "row_read_bytes": self.store.row_read_bytes,
                "peak_row_read_bytes": self.store.peak_row_read_bytes,
                "row_read_limit_bytes": MAX_ROW_READ_BYTES,
                "loaded_weight_bytes": self.loaded_bytes, "cache_hit_bytes": self.hit_bytes,
                "load_seconds": self.load_seconds, "retain_budget_bytes": self.retain_bytes,
                "retained_parameter_bytes": sum(self.specs[n].size for n in self.retained),
                "resident_retained_bytes": sum(self.specs[n].size for n in self.retained if n in self.cache)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handle in self.handles:
            handle.remove()
        for module, forward in self._patched_forwards:
            module.forward = forward
        for spec in self.specs.values():
            spec.module._parameters[spec.attribute] = spec.meta
        self.cache.clear()
        self.resident_bytes = 0
        self.store.close()


def attach_hooks(model, checkpoint_dir, device, cache_bytes, retain_bytes=0):
    """Attach to an already constructed model; caller owns nonparameter buffers."""
    return LazyCheckpoint(model, checkpoint_dir, device, cache_bytes, retain_bytes)


_construction_lock = threading.Lock()


@contextmanager
def _meta_parameter_factories():
    """Official parameters use empty/ones; zeros/full/arange initialize buffers.

    Construction is serialized because these Python factories are temporarily
    patched. Call before launching any inference threads. The hash-state cache
    uses empty too, and is materialized separately below before first use.
    """
    with _construction_lock:
        empty, ones = torch.empty, torch.ones

        def on_meta(factory):
            def allocate(*args, **kwargs):
                kwargs["device"] = "meta"
                return factory(*args, **kwargs)
            return allocate

        torch.empty, torch.ones = on_meta(empty), on_meta(ones)
        try:
            yield
        finally:
            torch.empty, torch.ones = empty, ones


def build_lazy_model(args, tokenizer, checkpoint_dir, device, cache_bytes, retain_bytes=0):
    """Construct the official Transformer without allocating its weight payload.

    The runner must import the official module defining ``type(args)`` first.
    Registered buffers retain their initialized values and move independently;
    model.to()/to_empty() must never be called on this 500+ GB model.
    """
    module = sys.modules[type(args).__module__]
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("lazy reference supports a single device only")
    # Buffer initialization must retain its CPU values even if the runner has
    # established a CUDA or meta default device around model construction.
    with torch.device("cpu"), _meta_parameter_factories():
        model = module.Transformer(args, tokenizer)
    moved = {}
    for owner in model.modules():
        for name, buffer in owner.named_buffers(recurse=False):
            key = id(buffer)
            if key not in moved:
                if buffer.is_meta:
                    # NgramHashState.cache is allocated with empty; consumed slots
                    # are written before reading. Zero initialization is deterministic.
                    moved[key] = torch.zeros(buffer.shape, dtype=buffer.dtype, device=device)
                else:
                    moved[key] = buffer.to(device)
            owner._buffers[name] = moved[key]
    model.eval()
    manager = attach_hooks(model, checkpoint_dir, device, cache_bytes, retain_bytes)
    return model, manager
