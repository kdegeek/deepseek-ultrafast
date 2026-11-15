//! Direct in-process C ABI. No model-server subprocess and no Python runtime.
use crate::{
    contract::{Generated, Usage},
    serving::{Config, Engine, Event, Job, RequestFailure},
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    os::windows::ffi::OsStrExt,
    path::Path,
    ptr,
    sync::Mutex,
};

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> Handle;
type Identity = unsafe extern "C" fn(Handle) -> *const c_char;
type Callback = unsafe extern "C" fn(*const c_char, *mut c_void) -> i32;
type Generate =
    unsafe extern "C" fn(Handle, *const c_char, Callback, *mut c_void, *mut c_char, usize) -> i32;
type Destroy = unsafe extern "C" fn(Handle);

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryExW(path: *const u16, file: Handle, flags: u32) -> Handle;
    fn GetProcAddress(module: Handle, name: *const u8) -> *mut c_void;
}

fn library(path: &Path) -> Result<Handle> {
    let path = path.canonicalize().context("resolve native library")?;
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // DLL_LOAD_DIR | DEFAULT_DIRS: dependencies come from this isolated build,
    // the executable, and Windows system directories; do not alter PATH.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), 0x100 | 0x1000) };
    ensure!(
        !module.is_null(),
        "load {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    // Deliberately retain DLLs for process lifetime. GGML registers global
    // backend callbacks; unloading code behind that registry is unsafe.
    Ok(module)
}
unsafe fn symbol(module: Handle, name: &[u8]) -> Result<*mut c_void> {
    let address = unsafe { GetProcAddress(module, name.as_ptr()) };
    ensure!(
        !address.is_null(),
        "native ABI symbol missing: {}",
        String::from_utf8_lossy(name)
    );
    Ok(address)
}
fn error_text(buffer: &[c_char]) -> String {
    // Buffer is zero-filled and native always receives its full bound.
    let bytes: Vec<u8> = buffer
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(crate) struct NativeEngine {
    handle: Handle,
    generate: Generate,
    destroy: Destroy,
    identity: Value,
}
impl NativeEngine {
    pub(crate) fn open(config: Config) -> Result<Box<dyn Engine>> {
        let module = library(&config.library)?;
        let (create, identity, generate, destroy): (Create, Identity, Generate, Destroy) = unsafe {
            (
                std::mem::transmute::<*mut c_void, Create>(symbol(module, b"qr_create\0")?),
                std::mem::transmute::<*mut c_void, Identity>(symbol(module, b"qr_identity\0")?),
                std::mem::transmute::<*mut c_void, Generate>(symbol(module, b"qr_generate\0")?),
                std::mem::transmute::<*mut c_void, Destroy>(symbol(module, b"qr_destroy\0")?),
            )
        };
        let input = CString::new(serde_json::to_string(&config)?)?;
        let mut error = vec![0; 4096];
        let handle = unsafe { create(input.as_ptr(), error.as_mut_ptr(), error.len()) };
        ensure!(
            !handle.is_null(),
            "initialize native Qwen: {}",
            error_text(&error)
        );
        let mut engine = Self {
            handle,
            generate,
            destroy,
            identity: Value::Null,
        };
        let identity = unsafe { identity(handle) };
        ensure!(!identity.is_null(), "native identity was null");
        engine.identity = serde_json::from_slice(unsafe { CStr::from_ptr(identity).to_bytes() })
            .context("native identity JSON")?;
        Ok(Box::new(engine))
    }
}
impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) };
    }
}

struct CallbackState<'a> {
    job: &'a Job,
    payload: Mutex<CallbackPayload>,
}
#[derive(Default)]
struct CallbackPayload {
    finished: Option<Generated>,
    emitted: String,
    error: Option<String>,
}
impl CallbackPayload {
    fn receive(&mut self, job: &Job, event: Value) -> Result<()> {
        match event["type"].as_str() {
            Some("poll") => (),
            Some("delta") => {
                ensure!(self.finished.is_none(), "native delta after finished event");
                let text = event["text"].as_str().context("native delta has no text")?;
                self.emitted.push_str(text);
                job.emit(Event::Delta(text.into()));
            }
            Some("finished") => {
                ensure!(self.finished.is_none(), "duplicate native finished event");
                let prompt_tokens = event["prompt_tokens"]
                    .as_u64()
                    .context("native prompt_tokens missing")?;
                let completion_tokens = event["completion_tokens"]
                    .as_u64()
                    .context("native completion_tokens missing")?;
                let reason = event["finish_reason"]
                    .as_str()
                    .context("native finish_reason missing")?;
                ensure!(
                    matches!(reason, "stop" | "length"),
                    "unsupported native finish_reason"
                );
                let text = event["text"]
                    .as_str()
                    .context("native final text missing")?;
                ensure!(
                    text == self.emitted,
                    "native streamed and final text diverged"
                );
                self.finished = Some(Generated {
                    text: text.into(),
                    finish_reason: reason.into(),
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens
                            .checked_add(completion_tokens)
                            .context("native usage overflow")?,
                    },
                    metrics: event["metrics"].clone(),
                });
            }
            _ => bail!("unknown native callback event"),
        }
        Ok(())
    }
}
unsafe extern "C" fn callback(input: *const c_char, user: *mut c_void) -> i32 {
    // Never unwind through C++, including on malformed native output.
    // GGML's CPU abort hook can run on a backend worker. Share only immutable
    // job data and mutex-protected callback state across those invocations.
    let state = unsafe { &*user.cast::<CallbackState<'_>>() };
    if state.job.cancelled() {
        return 0;
    }
    let mut payload = state.payload.lock().unwrap_or_else(|e| e.into_inner());
    if payload.error.is_some() {
        return 0;
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        ensure!(!input.is_null(), "null native event");
        let event = serde_json::from_slice(unsafe { CStr::from_ptr(input).to_bytes() })?;
        payload.receive(state.job, event)
    }));
    match outcome {
        Ok(Ok(())) => i32::from(!state.job.cancelled()),
        Ok(Err(error)) => {
            payload.error = Some(format!("{error:#}"));
            0
        }
        Err(_) => {
            payload.error = Some("native callback panicked".into());
            0
        }
    }
}
impl Engine for NativeEngine {
    fn identity(&self) -> Value {
        self.identity.clone()
    }
    fn generate(&mut self, job: &Job) -> Result<Option<Generated>> {
        let input = CString::new(serde_json::to_string(&job.request.native)?)?;
        let mut error = vec![0; 4096];
        let mut state = CallbackState {
            job,
            payload: Mutex::new(CallbackPayload::default()),
        };
        let code = unsafe {
            (self.generate)(
                self.handle,
                input.as_ptr(),
                callback,
                (&mut state as *mut CallbackState<'_>).cast(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let payload = state
            .payload
            .into_inner()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(error) = payload.error {
            bail!("native callback: {error}");
        }
        // Native returns only after clearing request KV/state and the sampler.
        // Cancellation never reopens admission before that cleanup completes.
        match code {
            0 if job.cancelled() => Ok(None),
            0 => payload
                .finished
                .map(Some)
                .context("native returned success without finished event"),
            1 => Ok(None),
            2 => Err(RequestFailure(error_text(&error)).into()),
            _ => bail!("native generation failed ({code}): {}", error_text(&error)),
        }
    }
}
