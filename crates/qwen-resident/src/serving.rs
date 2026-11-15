//! Qwen loopback API, following the fork's tested Strix single-owner lifecycle.
//! The native library owns GGUF tokenization, batched prefill, and sampling.
use crate::contract::{self as contract, Generated, MODEL, Request, Usage};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::to_bytes,
    extract::{Request as HttpRequest, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event as SseEvent, KeepAlive},
    },
    routing::{get, post},
};
use futures_core::Stream;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    task::{Context as TaskContext, Poll},
    thread::JoinHandle,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

const MAX_BODY: usize = 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(3600);
static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, serde::Serialize)]
pub struct Config {
    pub model_path: PathBuf,
    #[serde(skip)]
    pub library: PathBuf,
    #[serde(skip)]
    pub port: u16,
    pub context: usize,
    pub batch: usize,
    pub ubatch: usize,
    pub threads: usize,
    pub gpu_layers: usize,
    pub flash_attention: bool,
    pub kv_type: String,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.port != 0, "port must be nonzero");
        ensure!(
            (512..=32768).contains(&self.context),
            "context must be 512..=32768; only measured lengths are validated"
        );
        ensure!(
            (32..=8192).contains(&self.batch) && self.batch <= self.context,
            "batch must be 32..=8192 and at most context"
        );
        ensure!(
            self.ubatch >= 32 && self.ubatch <= self.batch,
            "ubatch must be 32..=batch"
        );
        ensure!((1..=32).contains(&self.threads), "threads must be 1..=32");
        ensure!(
            self.gpu_layers == 999,
            "gpu_layers must be 999 (full resident GPU offload)"
        );
        ensure!(
            matches!(self.kv_type.as_str(), "f16" | "q8_0"),
            "kv_type must be f16 or q8_0"
        );
        ensure!(
            self.flash_attention || self.kv_type == "f16",
            "quantized KV requires flash attention"
        );
        Ok(())
    }
}

pub(crate) enum Event {
    Delta(String),
    Finished(Generated),
    Error(String, bool),
}

#[derive(Debug)]
pub(crate) struct RequestFailure(pub(crate) String);
impl std::fmt::Display for RequestFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for RequestFailure {}

pub(crate) struct Job {
    pub(crate) request: Request,
    cancel: Arc<AtomicBool>,
    events: UnboundedSender<Event>,
    id: String,
}

impl Job {
    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire) || self.events.is_closed()
    }
    pub(crate) fn emit(&self, event: Event) {
        // At most max_tokens deltas plus a final flush and terminal event.
        // MAX_TOKENS bounds retained events even for a stalled TCP consumer.
        if !self.cancelled() {
            let _ = self.events.send(event);
        }
    }
}

pub(crate) trait Engine {
    fn identity(&self) -> Value;
    fn generate(&mut self, job: &Job) -> Result<Option<Generated>>;
}

struct Status {
    state: &'static str,
    failure: Option<String>,
    identity: Value,
    active: Option<Arc<AtomicBool>>,
    completed: u64,
    cancelled: u64,
    last_request: Value,
}

struct Worker {
    status: Arc<Mutex<Status>>,
    sender: mpsc::SyncSender<Job>,
    stopping: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start(factory: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static) -> Result<Self> {
        let status = Arc::new(Mutex::new(Status {
            state: "starting",
            failure: None,
            identity: Value::Null,
            active: None,
            completed: 0,
            cancelled: 0,
            last_request: Value::Null,
        }));
        let (sender, receiver) = mpsc::sync_channel::<Job>(1);
        let stopping = Arc::new(AtomicBool::new(false));
        let shared = status.clone();
        let stop = stopping.clone();
        let thread = std::thread::Builder::new().name("qwen-native-owner".into()).spawn(move || {
            // A Rust panic must not leave readiness stuck at ready/busy.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                let mut engine = factory()?;
                {
                    let mut state = shared.lock().expect("worker status");
                    state.identity = engine.identity();
                    if !stop.load(Ordering::Acquire) { state.state = "ready"; }
                }
                while !stop.load(Ordering::Acquire) {
                    let job = match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    };
                    let result = if job.cancelled() { Ok(None) } else { engine.generate(&job) };
                    // generate has cleared per-request state before admission reopens.
                    let mut state = shared.lock().expect("worker status");
                    state.active = None;
                    match result {
                        Ok(Some(result)) if !job.cancelled() => {
                            state.completed += 1;
                            state.last_request = json!({"id":job.id,"outcome":"completed","metrics":result.metrics});
                            drop(state);
                            job.emit(Event::Finished(result));
                        }
                        Ok(_) => {
                            state.cancelled += 1;
                            state.last_request = json!({"id":job.id,"outcome":"cancelled"});
                        }
                        Err(error) => {
                            let message = format!("{error:#}");
                            let recoverable = error.is::<RequestFailure>();
                            if !recoverable {
                                state.state = "failed";
                                state.failure = Some(message.clone());
                            }
                            state.last_request = json!({"id":job.id,"outcome":"failed"});
                            drop(state);
                            job.emit(Event::Error(message, recoverable));
                            if !recoverable { break; }
                        }
                    }
                }
                Ok(())
            }));
            if !matches!(run, Ok(Ok(()))) {
                let message = match run { Ok(Err(error)) => format!("{error:#}"), _ => "native worker panicked".into() };
                let mut state = shared.lock().unwrap_or_else(|e| e.into_inner());
                state.state = "failed";
                state.failure = Some(message);
                if let Some(cancel) = state.active.take() { cancel.store(true, Ordering::Release); }
            }
        })?;
        Ok(Self {
            status,
            sender,
            stopping,
            thread: Some(thread),
        })
    }

    fn submit(&self, job: Job) -> std::result::Result<(), StatusCode> {
        let mut state = self.status.lock().expect("worker status");
        if state.state != "ready" {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        if state.active.is_some() {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
        state.active = Some(job.cancel.clone());
        if self.sender.try_send(job).is_err() {
            state.active = None;
            state.state = "failed";
            state.failure = Some("GPU worker is unavailable".into());
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        Ok(())
    }

    fn snapshot(&self) -> Value {
        let state = self.status.lock().unwrap_or_else(|e| e.into_inner());
        json!({"engine":"native-rust-llama-cpp","experimental":true,"state":state.state,"pid":std::process::id(),"busy":state.active.is_some(),"queue_capacity":0,"error":state.failure,"identity":state.identity,"completed_requests":state.completed,"cancelled_requests":state.cancelled,"last_request":state.last_request})
    }

    fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
        let mut state = self.status.lock().unwrap_or_else(|e| e.into_inner());
        state.state = "stopping";
        if let Some(cancel) = &state.active {
            cancel.store(true, Ordering::Release);
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone)]
struct App {
    worker: Arc<Worker>,
    timeout: Duration,
}

fn error_value(message: impl AsRef<str>, kind: &str) -> Value {
    json!({"error":{"message":message.as_ref(),"type":kind,"param":null,"code":null}})
}
fn error(status: StatusCode, message: impl AsRef<str>) -> Response {
    let kind = if status == StatusCode::TOO_MANY_REQUESTS {
        "rate_limit_error"
    } else if status.is_server_error() {
        "engine_error"
    } else {
        "invalid_request_error"
    };
    let mut response = (status, Json(error_value(message, kind))).into_response();
    if matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
    ) {
        response
            .headers_mut()
            .insert("retry-after", "5".parse().expect("constant"));
    }
    response
}

fn local_headers(headers: &HeaderMap) -> bool {
    if headers.contains_key("origin") {
        return false;
    }
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    host == "localhost"
        || host == "127.0.0.1"
        || host.starts_with("localhost:")
        || host.starts_with("127.0.0.1:")
}

async fn local_only(request: HttpRequest, next: Next) -> Response {
    if !local_headers(request.headers()) {
        return error(
            StatusCode::FORBIDDEN,
            "only loopback Host names without browser Origin are accepted",
        );
    }
    next.run(request).await
}

async fn health(State(app): State<App>) -> Json<Value> {
    Json(app.worker.snapshot())
}
async fn ready(State(app): State<App>) -> Response {
    let state = app.worker.snapshot();
    (
        if state["state"] == "ready" {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(state),
    )
        .into_response()
}
async fn models() -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":MODEL,"object":"model","created":0,"owned_by":"Qwen"}]}),
    )
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

struct Metadata {
    id: String,
    created: u64,
    chat: bool,
    include_usage: bool,
}
impl Metadata {
    fn common(&self) -> Value {
        json!({"id":self.id,"created":self.created,"model":MODEL})
    }
    fn chunk(
        &self,
        text: Option<&str>,
        role: bool,
        reason: Option<&str>,
        usage: Option<&Usage>,
    ) -> Value {
        let mut out = self.common();
        out["object"] = json!(if self.chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        });
        let choice = if self.chat {
            let delta = if role {
                json!({"role":"assistant","content":""})
            } else if let Some(text) = text {
                json!({"content":text})
            } else {
                json!({})
            };
            json!({"index":0,"delta":delta,"finish_reason":reason})
        } else {
            json!({"index":0,"text":text.unwrap_or(""),"logprobs":null,"finish_reason":reason})
        };
        out["choices"] = if usage.is_some() {
            json!([])
        } else {
            json!([choice])
        };
        if self.include_usage {
            out["usage"] = json!(usage);
        }
        out
    }
    fn result(&self, result: &Generated) -> Value {
        let mut out = self.common();
        out["object"] = json!(if self.chat {
            "chat.completion"
        } else {
            "text_completion"
        });
        out["choices"] = if self.chat {
            json!([{"index":0,"message":{"role":"assistant","content":result.text},"finish_reason":result.finish_reason}])
        } else {
            json!([{"index":0,"text":result.text,"logprobs":null,"finish_reason":result.finish_reason}])
        };
        out["usage"] = json!(result.usage);
        out["metrics"] = result.metrics.clone();
        out
    }
}

fn sse(value: Value) -> SseEvent {
    SseEvent::default().data(value.to_string())
}

struct EventStream {
    receiver: UnboundedReceiver<Event>,
    guard: CancelOnDrop,
    metadata: Metadata,
    pending: VecDeque<SseEvent>,
    deadline: Pin<Box<tokio::time::Sleep>>,
    finished: bool,
    first_event: Option<Event>,
}
impl Stream for EventStream {
    type Item = std::result::Result<SseEvent, Infallible>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            if this.deadline.as_mut().poll(cx).is_ready() {
                this.guard.0.store(true, Ordering::Release);
                this.pending.push_back(sse(error_value(
                    "request deadline exceeded",
                    "engine_error",
                )));
                this.pending.push_back(SseEvent::default().data("[DONE]"));
                this.finished = true;
                continue;
            }
            let next = match this.first_event.take() {
                Some(event) => Poll::Ready(Some(event)),
                None => this.receiver.poll_recv(cx),
            };
            match next {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Event::Delta(text))) => {
                    return Poll::Ready(Some(Ok(sse(this.metadata.chunk(
                        Some(&text),
                        false,
                        None,
                        None,
                    )))));
                }
                Poll::Ready(Some(Event::Finished(result))) => {
                    let mut terminal =
                        this.metadata
                            .chunk(None, false, Some(&result.finish_reason), None);
                    terminal["metrics"] = result.metrics.clone();
                    this.pending.push_back(sse(terminal));
                    if this.metadata.include_usage {
                        this.pending.push_back(sse(this.metadata.chunk(
                            None,
                            false,
                            None,
                            Some(&result.usage),
                        )));
                    }
                    this.pending.push_back(SseEvent::default().data("[DONE]"));
                    this.finished = true;
                }
                Poll::Ready(event) => {
                    let message = match event {
                        Some(Event::Error(message, _)) => message,
                        _ => "native worker stopped before completing the request".into(),
                    };
                    this.pending
                        .push_back(sse(error_value(message, "engine_error")));
                    this.pending.push_back(SseEvent::default().data("[DONE]"));
                    this.finished = true;
                }
            }
        }
    }
}

async fn completions(State(app): State<App>, request: HttpRequest) -> Response {
    generate(app, request, false).await
}
async fn chat_completions(State(app): State<App>, request: HttpRequest) -> Response {
    generate(app, request, true).await
}

async fn generate(app: App, request: HttpRequest, chat: bool) -> Response {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    if !content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .eq_ignore_ascii_case("application/json")
    {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    if app.worker.snapshot()["busy"] == true {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "one inference is active; no waiting queue",
        );
    }
    let bytes = match tokio::time::timeout(
        Duration::from_secs(10),
        to_bytes(request.into_body(), MAX_BODY),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => {
            return error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds 1 MiB or could not be read",
            );
        }
        Err(_) => return error(StatusCode::REQUEST_TIMEOUT, "request body timed out"),
    };
    let body = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let request = match contract::validate(body, chat) {
        Ok(request) => request,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    // Context checks use the native GGUF tokenizer before decode.
    let serial = NEXT_REQUEST.fetch_add(1, Ordering::Relaxed);
    let metadata = Metadata {
        id: format!(
            "{}cmpl-{}-{serial}",
            if chat { "chat" } else { "" },
            std::process::id()
        ),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        chat,
        include_usage: request.include_usage,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let (events, mut receiver) = unbounded_channel();
    let streaming = request.stream;
    let job = Job {
        request,
        cancel,
        events,
        id: metadata.id.clone(),
    };
    if let Err(status) = app.worker.submit(job) {
        return error(
            status,
            if status == StatusCode::TOO_MANY_REQUESTS {
                "engine busy"
            } else {
                "engine not ready"
            },
        );
    }
    if streaming {
        // Native tokenization and context validation precede HTTP 200 headers.
        // Poll callbacks do not emit bytes, so errors before the first token
        // keep their real HTTP status instead of becoming a successful SSE.
        let deadline = tokio::time::Instant::now() + app.timeout;
        let first_event = match tokio::time::timeout_at(deadline, receiver.recv()).await {
            Ok(Some(Event::Error(message, recoverable))) => {
                return error(
                    if recoverable {
                        StatusCode::BAD_REQUEST
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    },
                    message,
                );
            }
            Ok(Some(event)) => event,
            Ok(None) => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "native worker stopped before first event",
                );
            }
            Err(_) => return error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
        };
        let mut pending = VecDeque::new();
        if chat {
            pending.push_back(sse(metadata.chunk(None, true, None, None)));
        }
        let stream = EventStream {
            receiver,
            guard,
            metadata,
            pending,
            deadline: Box::pin(tokio::time::sleep_until(deadline)),
            finished: false,
            first_event: Some(first_event),
        };
        return Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(10))
                    .text("working"),
            )
            .into_response();
    }
    let result = tokio::time::timeout(app.timeout, async {
        loop {
            match receiver.recv().await {
                Some(Event::Delta(_)) => (),
                Some(Event::Finished(result)) => return Ok(result),
                Some(Event::Error(message, recoverable)) => return Err((message, recoverable)),
                None => {
                    return Err((
                        "native worker stopped before completing the request".into(),
                        false,
                    ));
                }
            }
        }
    })
    .await;
    drop(guard);
    match result {
        Ok(Ok(result)) => Json(metadata.result(&result)).into_response(),
        Ok(Err((message, recoverable))) => error(
            if recoverable {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            },
            message,
        ),
        Err(_) => error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    }
}

fn router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "unknown endpoint") })
        .layer(middleware::from_fn(local_only))
        .with_state(app)
}

pub fn run(config: Config) -> Result<()> {
    config.validate()?;
    #[cfg(not(windows))]
    anyhow::bail!("native Qwen serving currently requires Windows");
    #[cfg(windows)]
    {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,config.port)).await.context("loopback listener")?;
            let engine_config = config.clone();
            let worker = Arc::new(Worker::start(move || crate::native::NativeEngine::open(engine_config))?);
            let app = App {worker:worker.clone(),timeout:REQUEST_TIMEOUT};
            eprintln!("Qwen native API: http://127.0.0.1:{}/v1; /readyz reports model loading; context {}",config.port,config.context);
            let signal_worker = worker.clone();
            let result = axum::serve(listener,router(app)).with_graceful_shutdown(async move {
                let _ = tokio::signal::ctrl_c().await;
                signal_worker.stop();
            }).await;
            worker.stop();
            tokio::task::spawn_blocking(move || drop(worker)).await?;
            result?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request as Http};
    use std::sync::atomic::AtomicUsize;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_stream::StreamExt;
    use tower::ServiceExt;

    #[derive(Default)]
    struct Fixture {
        entered: AtomicUsize,
        cleaned: AtomicUsize,
        finish: AtomicBool,
    }
    struct FakeEngine(Arc<Fixture>);
    impl Engine for FakeEngine {
        fn identity(&self) -> Value {
            json!({"fixture":true})
        }
        fn generate(&mut self, job: &Job) -> Result<Option<Generated>> {
            self.0.entered.fetch_add(1, Ordering::Release);
            let result = (|| {
                if job.request.native["prompt"] == "failure" {
                    anyhow::bail!("fixture failure");
                }
                if job.request.native["prompt"] == "request_failure" {
                    return Err(RequestFailure("unsupported model control output".into()).into());
                }
                if job.request.native["prompt"] != "prefill" {
                    job.emit(Event::Delta("first".into()));
                }
                while !self.0.finish.load(Ordering::Acquire) {
                    if job.cancelled() {
                        return Ok(None);
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                if job.cancelled() {
                    return Ok(None);
                }
                if job.request.native["prompt"] == "prefill" {
                    job.emit(Event::Delta("first".into()));
                }
                job.emit(Event::Delta(" second".into()));
                Ok(Some(Generated {
                    text: "first second".into(),
                    finish_reason: "stop".into(),
                    usage: Usage {
                        prompt_tokens: 1,
                        completion_tokens: 3,
                        total_tokens: 4,
                    },
                    metrics: json!({"fixture":true}),
                }))
            })();
            self.0.cleaned.fetch_add(1, Ordering::Release);
            result
        }
    }
    async fn until(mut predicate: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(4), async {
            while !predicate() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("fixture condition timed out");
    }
    async fn setup() -> (App, Arc<Fixture>) {
        let fixture = Arc::new(Fixture::default());
        let owned = fixture.clone();
        let worker = Arc::new(Worker::start(move || Ok(Box::new(FakeEngine(owned)))).unwrap());
        until(|| worker.snapshot()["state"] == "ready").await;
        (
            App {
                worker,
                timeout: Duration::from_secs(4),
            },
            fixture,
        )
    }
    fn request(path: &str, body: Value) -> Http<Body> {
        Http::builder()
            .method("POST")
            .uri(path)
            .header("host", "127.0.0.1:18081")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }
    fn get(path: &str) -> Http<Body> {
        Http::builder()
            .uri(path)
            .header("host", "127.0.0.1:18081")
            .body(Body::empty())
            .unwrap()
    }
    fn base() -> Value {
        json!({"model":MODEL,"prompt":"Hi","max_tokens":3})
    }
    async fn body(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), MAX_BODY * 2).await.unwrap())
            .unwrap()
    }

    #[tokio::test]
    async fn http_contract_admission_and_limits() {
        let (app, fixture) = setup().await;
        let service = router(app.clone());
        let models = service.clone().oneshot(get("/v1/models")).await.unwrap();
        assert_eq!(models.status(), StatusCode::OK);
        assert_eq!(body(models).await["data"][0]["id"], MODEL);
        for (mut req, expected) in [
            (
                request("/v1/completions", json!([])),
                StatusCode::BAD_REQUEST,
            ),
            (request("/missing", base()), StatusCode::NOT_FOUND),
            (
                request(
                    "/v1/completions",
                    json!({"model":MODEL,"prompt":"x".repeat(MAX_BODY)}),
                ),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            req.headers_mut()
                .insert("host", "localhost:18081".parse().unwrap());
            let response = service.clone().oneshot(req).await.unwrap();
            assert_eq!(response.status(), expected);
            assert!(body(response).await.get("error").is_some());
        }
        let mut req = request("/v1/completions", base());
        req.headers_mut()
            .insert("origin", "http://127.0.0.1".parse().unwrap());
        assert_eq!(
            service.clone().oneshot(req).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut req = request("/v1/completions", base());
        req.headers_mut()
            .insert("host", "evil.example:18081".parse().unwrap());
        assert_eq!(
            service.clone().oneshot(req).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let mut req = request("/v1/completions", base());
        req.headers_mut().remove("content-type");
        assert_eq!(
            service.oneshot(req).await.unwrap().status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE
        );
        assert_eq!(fixture.entered.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn incremental_sse_busy_usage_and_isolation() {
        let (app, fixture) = setup().await;
        let service = router(app.clone());
        let response=service.clone().oneshot(request("/v1/completions",json!({"model":MODEL,"prompt":"Hi","stream":true,"stream_options":{"include_usage":true}}))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.into_body().into_data_stream();
        let first = stream.next().await.unwrap().unwrap();
        assert!(String::from_utf8_lossy(&first).contains("first"));
        assert_eq!(fixture.cleaned.load(Ordering::Acquire), 0);
        let health = body(service.clone().oneshot(get("/health")).await.unwrap()).await;
        assert_eq!(health["busy"], true);
        assert_eq!(
            service
                .clone()
                .oneshot(request("/v1/completions", base()))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        fixture.finish.store(true, Ordering::Release);
        let mut rest = Vec::new();
        while let Some(chunk) = stream.next().await {
            rest.extend_from_slice(&chunk.unwrap());
        }
        let rest = String::from_utf8(rest).unwrap();
        assert!(rest.contains(" second"));
        assert!(rest.contains("\"completion_tokens\":3"));
        assert!(rest.contains("\"choices\":[]"));
        assert_eq!(rest.matches("[DONE]").count(), 1);
        until(|| app.worker.snapshot()["busy"] == false).await;
        let response = service
            .oneshot(request(
                "/v1/chat/completions",
                json!({"model":MODEL,"messages":[{"role":"user","content":"Hi"}]}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body(response).await["choices"][0]["message"]["content"],
            "first second"
        );
        assert_eq!(fixture.cleaned.load(Ordering::Acquire), 2);
    }

    #[tokio::test]
    async fn dropped_stream_cancels_before_readmission() {
        let (app, fixture) = setup().await;
        let service = router(app.clone());
        let response = service
            .clone()
            .oneshot(request(
                "/v1/completions",
                json!({"model":MODEL,"prompt":"Hi","stream":true}),
            ))
            .await
            .unwrap();
        until(|| fixture.entered.load(Ordering::Acquire) == 1).await;
        drop(response);
        until(|| app.worker.snapshot()["cancelled_requests"] == 1).await;
        assert_eq!(fixture.cleaned.load(Ordering::Acquire), 1);
        assert_eq!(app.worker.snapshot()["busy"], false);
        fixture.finish.store(true, Ordering::Release);
        assert_eq!(
            service
                .oneshot(request("/v1/completions", base()))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn engine_failure_is_closed_and_readiness_reports_it() {
        let (app, _fixture) = setup().await;
        let service = router(app.clone());
        let response = service
            .clone()
            .oneshot(request(
                "/v1/completions",
                json!({"model":MODEL,"prompt":"failure"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(app.worker.snapshot()["state"], "failed");
        assert_eq!(
            service
                .clone()
                .oneshot(get("/readyz"))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            service
                .oneshot(request("/v1/completions", base()))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn request_output_error_cleans_up_and_allows_next_request() {
        let (app, fixture) = setup().await;
        let service = router(app.clone());
        let response = service
            .clone()
            .oneshot(request(
                "/v1/completions",
                json!({"model":MODEL,"prompt":"request_failure"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(app.worker.snapshot()["state"], "ready");
        assert_eq!(app.worker.snapshot()["busy"], false);
        assert_eq!(fixture.cleaned.load(Ordering::Acquire), 1);
        fixture.finish.store(true, Ordering::Release);
        let response = service
            .oneshot(request("/v1/completions", base()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body(response).await["choices"][0]["text"], "first second");
    }

    #[tokio::test]
    async fn streaming_native_validation_error_keeps_http_400() {
        let (app, fixture) = setup().await;
        let response = router(app.clone())
            .oneshot(request(
                "/v1/completions",
                json!({"model":MODEL,"prompt":"request_failure","stream":true}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(body(response).await.get("error").is_some());
        assert_eq!(fixture.cleaned.load(Ordering::Acquire), 1);
        assert_eq!(app.worker.snapshot()["busy"], false);
    }

    #[tokio::test]
    async fn initialization_failure_remains_unready() {
        let worker = Arc::new(Worker::start(|| anyhow::bail!("fixture load failure")).unwrap());
        until(|| worker.snapshot()["state"] == "failed").await;
        let app = App {
            worker,
            timeout: Duration::from_secs(1),
        };
        let response = router(app.clone()).oneshot(get("/readyz")).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            body(response).await["error"]
                .as_str()
                .unwrap()
                .contains("load failure")
        );
    }

    #[tokio::test]
    async fn deadline_cancels_worker() {
        let (mut app, fixture) = setup().await;
        app.timeout = Duration::from_millis(30);
        let response = router(app.clone())
            .oneshot(request("/v1/completions", base()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        until(|| fixture.cleaned.load(Ordering::Acquire) == 1).await;
        assert_eq!(app.worker.snapshot()["state"], "ready");
    }

    #[tokio::test]
    async fn real_tcp_disconnect_cancels_stream_and_nonstream() {
        let (app, fixture) = setup().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, stopped) = tokio::sync::oneshot::channel();
        let service = router(app.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, service)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        for (index, (streaming, prompt)) in [(true, "Hi"), (false, "Hi"), (true, "prefill")]
            .into_iter()
            .enumerate()
        {
            let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
            let payload = json!({"model":MODEL,"prompt":prompt,"stream":streaming}).to_string();
            let raw = format!(
                "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
                payload.len()
            );
            client.write_all(raw.as_bytes()).await.unwrap();
            until(|| fixture.entered.load(Ordering::Acquire) == index + 1).await;
            if streaming && prompt != "prefill" {
                let mut bytes = [0; 4096];
                let n = tokio::time::timeout(Duration::from_secs(2), client.read(&mut bytes))
                    .await
                    .unwrap()
                    .unwrap();
                assert!(String::from_utf8_lossy(&bytes[..n]).contains("200 OK"));
            }
            drop(client);
            until(|| app.worker.snapshot()["cancelled_requests"] == index + 1).await;
            assert_eq!(fixture.cleaned.load(Ordering::Acquire), index + 1);
        }
        let _ = shutdown.send(());
        server.await.unwrap();
    }
}
