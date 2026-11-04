//! Serves the Cursor local-loop RunInference stream.
//!
//! The local loop's first append is a RunInferenceClientMessage. Its run
//! request carries a conversation id string, not a ConversationStateStructure.
//! Replies on the open RunInferenceSSE stream are RunInferenceServerMessage
//! frames: run_ready, then invocation text, then invocation_end.

use std::{collections::HashMap, io::Read, sync::OnceLock, time::Duration};

use flate2::read::GzDecoder;
use futures_util::StreamExt;
use prost::Message;
use prost_types::value::Kind;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::{
    cursor::{
        protocol::{
            connect,
            proto::inference::v1::{
                inference_content_part, inference_core_message, inference_stream_response,
                run_inference_client_message, run_inference_server_message, InferenceContentPart,
                InferenceCoreMessage, InferenceStreamRequest, InferenceStreamResponse,
                InferenceTextStreamPart, InferenceThinkingStreamPart,
                RunInferenceCompressedRequest, RunInferenceHeartbeat, RunInferenceInvocationEnd,
                RunInferenceInvocationError, RunInferenceInvocationResponse,
                RunInferenceInvokeModel, RunInferenceRoutingMessage, RunInferenceRunReady,
                RunInferenceRunRequest, RunInferenceServerMessage,
            },
        },
        transport::{TransportHandle, TransportParent, TransportRegistry},
    },
    model::{
        ContentPart, ModelInvocation, ModelRequest, ModelSpec, ProjectedContent, ProjectedMessage,
        PromptSpec, Role,
    },
    provider::ModelEvent,
    Error, Result,
};

const ROLE_USER: i32 = 1;
const ROLE_ASSISTANT: i32 = 2;
const ROLE_SYSTEM: i32 = 4;
const ENCODING_ZSTD: i32 = 1;
const ENCODING_GZIP: i32 = 2;
const MAX_INFERENCE_BYTES: usize = 8 * 1024 * 1024;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

struct Session {
    model_id: String,
    routing_text: String,
    generation: CancellationToken,
    heartbeats: CancellationToken,
}

impl Session {
    fn new(model_id: String, routing_text: String) -> Self {
        Self {
            model_id,
            routing_text,
            generation: CancellationToken::new(),
            heartbeats: CancellationToken::new(),
        }
    }
}

fn sessions() -> &'static Mutex<HashMap<String, Session>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub async fn accept(
    registry: &TransportRegistry,
    request_id: &str,
    seqno: i64,
    message: run_inference_client_message::Message,
    parent: Option<TransportParent>,
) -> Result<()> {
    let handle = registry
        .get_or_create_for_append(request_id, model_of(&message).is_some())
        .await?;
    let _admission = handle.admit()?;
    if let Some(conversation_id) = conversation_of(&message) {
        handle.set_conversation_id(conversation_id)?;
    }
    if let Some(parent) = parent {
        handle.set_parent(parent)?;
    }
    tracing::info!(
        request_id,
        seqno,
        kind = message_kind(&message),
        "accepted local inference append"
    );
    match message {
        run_inference_client_message::Message::RunRequest(request) => {
            open_run(&handle, request).await
        }
        run_inference_client_message::Message::InvokeModel(invoke) => {
            start_invocation(registry, &handle, invoke).await
        }
        run_inference_client_message::Message::ClientHeartbeat(_) => emit_heartbeat(&handle),
        run_inference_client_message::Message::CancelInvocation(_) => {
            cancel_generation(request_id).await;
            Ok(())
        }
        run_inference_client_message::Message::FinishRun(_) => {
            finish_session(request_id).await;
            Ok(())
        }
    }
}

pub fn project_prompt(
    request: &InferenceStreamRequest,
    routing_fallback: &str,
) -> Result<ModelRequest> {
    let mut instructions = String::new();
    let mut history = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        let text = message_text(message);
        if text.is_empty() {
            continue;
        }
        match message.role {
            ROLE_SYSTEM => {
                if !instructions.is_empty() {
                    instructions.push('\n');
                }
                instructions.push_str(&text);
            }
            ROLE_ASSISTANT => history.push(ProjectedMessage {
                message_id: format!("inference-{index}"),
                role: Role::Assistant,
                content: ProjectedContent::Assistant {
                    text,
                    thinking: String::new(),
                    replay_state: None,
                    calls: Vec::new(),
                },
            }),
            _ => history.push(ProjectedMessage {
                message_id: format!("inference-{index}"),
                role: Role::User,
                content: ProjectedContent::Parts(vec![ContentPart::Text { text }]),
            }),
        }
    }
    if !history
        .iter()
        .any(|message| matches!(message.role, Role::User | Role::Assistant))
        && !routing_fallback.is_empty()
    {
        history.push(ProjectedMessage {
            message_id: "inference-routing".into(),
            role: Role::User,
            content: ProjectedContent::Parts(vec![ContentPart::Text {
                text: routing_fallback.to_owned(),
            }]),
        });
    }
    if history.is_empty() {
        return Err(Error::Protocol("inference request has no user text".into()));
    }
    if instructions.is_empty() {
        instructions = "Reply to the latest user message.".into();
    }
    let model_id = request_model_id(request);
    Ok(ModelRequest {
        prompt: PromptSpec {
            instructions,
            tools: Vec::new(),
        },
        model: ModelSpec::new(model_id),
        history,
    })
}

fn message_text(message: &InferenceCoreMessage) -> String {
    match &message.content {
        Some(inference_core_message::Content::Text(text)) => text.clone(),
        Some(inference_core_message::Content::Parts(parts)) => parts
            .parts
            .iter()
            .filter_map(part_text)
            .collect::<Vec<_>>()
            .join("\n"),
        Some(inference_core_message::Content::ToolContent(content)) => content
            .parts
            .iter()
            .map(|part| {
                let body = value_text(&part.result);
                if part.tool_name.is_empty() {
                    body
                } else if body.is_empty() {
                    format!("Tool result ({})", part.tool_name)
                } else {
                    format!("Tool result ({}): {body}", part.tool_name)
                }
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

fn part_text(part: &InferenceContentPart) -> Option<String> {
    match &part.part {
        Some(inference_content_part::Part::Text(text)) if !text.text.is_empty() => {
            Some(text.text.clone())
        }
        Some(inference_content_part::Part::Image(_)) => {
            Some("（这轮附有图片，本地循环只把文字交给模型）".into())
        }
        Some(inference_content_part::Part::File(file)) => Some(format!(
            "（这轮附有文件 {}，本地循环只把文字交给模型）",
            file.filename
        )),
        _ => None,
    }
}

fn value_text(bytes: &[u8]) -> String {
    let Ok(value) = prost_types::Value::decode(bytes) else {
        return String::new();
    };
    kind_text(value.kind.as_ref(), 0)
}

fn kind_text(kind: Option<&Kind>, depth: usize) -> String {
    if depth > 4 {
        return String::new();
    }
    match kind {
        Some(Kind::StringValue(text)) => text.clone(),
        Some(Kind::NumberValue(value)) => value.to_string(),
        Some(Kind::BoolValue(value)) => value.to_string(),
        Some(Kind::StructValue(fields)) => fields
            .fields
            .iter()
            .map(|(key, value)| format!("{key}: {}", kind_text(value.kind.as_ref(), depth + 1)))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Kind::ListValue(values)) => values
            .values
            .iter()
            .map(|value| kind_text(value.kind.as_ref(), depth + 1))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

async fn open_run(handle: &TransportHandle, request: RunInferenceRunRequest) -> Result<()> {
    let model = request.requested_model.clone().unwrap_or_default();
    let routing_text = routing_user_text(&request.routing_conversation);
    let mut table = sessions().lock().await;
    if let Some(previous) = table.remove(handle.request_id()) {
        previous.heartbeats.cancel();
        previous.generation.cancel();
    }
    let session = Session::new(model.model_id.clone(), routing_text);
    let heartbeats = session.heartbeats.clone();
    table.insert(handle.request_id().to_owned(), session);
    drop(table);
    spawn_heartbeats(handle.clone(), heartbeats);
    emit_heartbeat(handle)?;
    emit(
        handle,
        &server(run_inference_server_message::Message::RunReady(
            RunInferenceRunReady {
                resolved_model: Some(model),
                accepts_client_heartbeat: true,
            },
        )),
    )
}

async fn start_invocation(
    registry: &TransportRegistry,
    handle: &TransportHandle,
    invoke: RunInferenceInvokeModel,
) -> Result<()> {
    let request = stream_request(&invoke)?;
    let invocation_id = invoke.invocation_id.clone();
    let (model_id, routing_text, cancel, heartbeats) = {
        let mut table = sessions().lock().await;
        let created = !table.contains_key(handle.request_id());
        let session = table
            .entry(handle.request_id().to_owned())
            .or_insert_with(|| Session::new(String::new(), String::new()));
        if session.model_id.is_empty() {
            session.model_id = request_model_id(&request);
        }
        session.generation.cancel();
        session.generation = CancellationToken::new();
        (
            session.model_id.clone(),
            session.routing_text.clone(),
            session.generation.clone(),
            created.then(|| session.heartbeats.clone()),
        )
    };
    if let Some(heartbeats) = heartbeats {
        spawn_heartbeats(handle.clone(), heartbeats);
    }
    let mut projected = project_prompt(&request, &routing_text)?;
    if projected.model.model_id.is_empty() {
        projected.model.model_id = model_id;
    }
    if projected.model.model_id.is_empty() {
        return Err(Error::Protocol(
            "inference invocation has no model id".into(),
        ));
    }
    let provider = registry.conversations().dependencies().provider.clone();
    let handle = handle.clone();
    let call_id = if invocation_id.is_empty() {
        format!("inference-{}", handle.request_id())
    } else {
        invocation_id.clone()
    };
    let conversation_id = if request.conversation_id.is_empty() {
        handle.request_id().to_owned()
    } else {
        request.conversation_id.clone()
    };
    tokio::spawn(async move {
        let invocation = ModelInvocation {
            call_id: call_id.clone(),
            run_id: call_id,
            conversation_id,
            provider_call_index: 0,
            request: projected,
        };
        if let Err(error) =
            stream_invocation(&handle, &invocation_id, provider, invocation, cancel).await
        {
            tracing::warn!(
                request_id = handle.request_id(),
                %error,
                "inference invocation failed"
            );
            let _ = emit_end(&handle, &invocation_id, Some(error.to_string()));
        }
    });
    Ok(())
}

async fn stream_invocation(
    handle: &TransportHandle,
    invocation_id: &str,
    provider: std::sync::Arc<dyn crate::provider::Provider>,
    invocation: ModelInvocation,
    cancel: CancellationToken,
) -> Result<()> {
    let mut stream = provider.stream(invocation, cancel.clone());
    let mut saw_text = false;
    let mut tool_names = Vec::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                emit_end(handle, invocation_id, None)?;
                return Ok(());
            }
            event = stream.next() => {
                let Some(event) = event else {
                    break;
                };
                match event? {
                    ModelEvent::TextDelta(text) if !text.is_empty() => {
                        saw_text = true;
                        emit_text(handle, invocation_id, &text, false)?;
                    }
                    ModelEvent::ThinkingDelta(text) if !text.is_empty() => {
                        emit_thinking(handle, invocation_id, &text)?;
                    }
                    ModelEvent::ToolCallStart { name, .. } => {
                        if !name.is_empty() {
                            tool_names.push(name);
                        }
                    }
                    ModelEvent::Done(_) => break,
                    _ => {}
                }
            }
        }
    }
    if !tool_names.is_empty() {
        emit_text(
            handle,
            invocation_id,
            &format!(
                "Tool calls are not executed on this local inference loop: {}.",
                tool_names.join(", ")
            ),
            false,
        )?;
        saw_text = true;
    }
    if !saw_text {
        return Err(Error::Provider(
            "inference provider returned no text".into(),
        ));
    }
    emit_text(handle, invocation_id, "", true)?;
    emit_end(handle, invocation_id, None)
}

fn stream_request(invoke: &RunInferenceInvokeModel) -> Result<InferenceStreamRequest> {
    if let Some(request) = &invoke.request {
        return Ok(request.clone());
    }
    let Some(compressed) = &invoke.compressed_request else {
        return Err(Error::Protocol(
            "inference invoke_model has no request".into(),
        ));
    };
    let bytes = inflate(compressed)?;
    InferenceStreamRequest::decode(bytes.as_slice()).map_err(|error| {
        Error::Protocol(format!(
            "compressed inference request did not decode: {error}"
        ))
    })
}

fn inflate(compressed: &RunInferenceCompressedRequest) -> Result<Vec<u8>> {
    let limit = match usize::try_from(compressed.decoded_size) {
        Ok(0) | Err(_) => MAX_INFERENCE_BYTES,
        Ok(size) => size.min(MAX_INFERENCE_BYTES),
    };
    if compressed.encoding == ENCODING_ZSTD {
        return Err(Error::Protocol(
            "inference request compression is zstd; gzip and uncompressed requests are accepted"
                .into(),
        ));
    }
    if compressed.encoding != ENCODING_GZIP {
        if compressed.data.len() > limit {
            return Err(Error::Protocol(
                "inference request exceeds the size limit".into(),
            ));
        }
        return Ok(compressed.data.clone());
    }
    let mut decoder = GzDecoder::new(compressed.data.as_slice());
    let mut output = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = decoder.read(&mut buffer).map_err(|error| {
            Error::Protocol(format!("inference gzip request could not be read: {error}"))
        })?;
        if read == 0 {
            break;
        }
        if output.len().saturating_add(read) > limit {
            return Err(Error::Protocol(
                "inference request exceeds the size limit".into(),
            ));
        }
        output.extend_from_slice(&buffer[..read]);
    }
    Ok(output)
}

fn routing_user_text(routing: &[RunInferenceRoutingMessage]) -> String {
    let user_text = routing
        .iter()
        .filter(|message| message.role == ROLE_USER && !message.text.is_empty())
        .map(|message| message.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if !user_text.is_empty() {
        return user_text;
    }
    routing
        .iter()
        .filter(|message| !message.text.is_empty())
        .map(|message| message.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn request_model_id(request: &InferenceStreamRequest) -> String {
    if !request.model_id.is_empty() {
        return request.model_id.clone();
    }
    request
        .requested_model
        .as_ref()
        .map(|model| model.model_id.clone())
        .filter(|model| !model.is_empty())
        .unwrap_or_default()
}

fn model_of(message: &run_inference_client_message::Message) -> Option<&str> {
    let model_id = match message {
        run_inference_client_message::Message::RunRequest(request) => request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str()),
        run_inference_client_message::Message::InvokeModel(invoke) => {
            let request = invoke.request.as_ref()?;
            if !request.model_id.is_empty() {
                return Some(request.model_id.as_str());
            }
            request
                .requested_model
                .as_ref()
                .map(|model| model.model_id.as_str())
        }
        _ => None,
    };
    model_id.filter(|model| !model.is_empty())
}

fn conversation_of(message: &run_inference_client_message::Message) -> Option<&str> {
    let conversation_id = match message {
        run_inference_client_message::Message::RunRequest(request) => {
            request.conversation_id.as_str()
        }
        run_inference_client_message::Message::InvokeModel(invoke) => invoke
            .request
            .as_ref()
            .map(|request| request.conversation_id.as_str())
            .unwrap_or(""),
        _ => "",
    };
    if conversation_id.is_empty() {
        None
    } else {
        Some(conversation_id)
    }
}

fn message_kind(message: &run_inference_client_message::Message) -> &'static str {
    match message {
        run_inference_client_message::Message::RunRequest(_) => "run_request",
        run_inference_client_message::Message::InvokeModel(_) => "invoke_model",
        run_inference_client_message::Message::CancelInvocation(_) => "cancel_invocation",
        run_inference_client_message::Message::FinishRun(_) => "finish_run",
        run_inference_client_message::Message::ClientHeartbeat(_) => "client_heartbeat",
    }
}

async fn cancel_generation(request_id: &str) {
    if let Some(session) = sessions().lock().await.get(request_id) {
        session.generation.cancel();
    }
}

async fn finish_session(request_id: &str) {
    if let Some(session) = sessions().lock().await.remove(request_id) {
        session.heartbeats.cancel();
        session.generation.cancel();
    }
}

fn spawn_heartbeats(handle: TransportHandle, cancel: CancellationToken) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = interval.tick() => {
                    if emit_heartbeat(&handle).is_err() {
                        break;
                    }
                }
            }
        }
    });
}

fn emit_heartbeat(handle: &TransportHandle) -> Result<()> {
    emit(
        handle,
        &server(run_inference_server_message::Message::Heartbeat(
            RunInferenceHeartbeat {
                accepts_speculative_invocations: false,
            },
        )),
    )
}

fn emit_text(
    handle: &TransportHandle,
    invocation_id: &str,
    text: &str,
    is_final: bool,
) -> Result<()> {
    emit(
        handle,
        &server(run_inference_server_message::Message::InvocationResponse(
            RunInferenceInvocationResponse {
                invocation_id: invocation_id.to_owned(),
                response: Some(InferenceStreamResponse {
                    response: Some(inference_stream_response::Response::TextPart(
                        InferenceTextStreamPart {
                            text: text.to_owned(),
                            is_final,
                        },
                    )),
                }),
            },
        )),
    )
}

fn emit_thinking(handle: &TransportHandle, invocation_id: &str, text: &str) -> Result<()> {
    emit(
        handle,
        &server(run_inference_server_message::Message::InvocationResponse(
            RunInferenceInvocationResponse {
                invocation_id: invocation_id.to_owned(),
                response: Some(InferenceStreamResponse {
                    response: Some(inference_stream_response::Response::ThinkingPart(
                        InferenceThinkingStreamPart {
                            text: text.to_owned(),
                            is_final: false,
                        },
                    )),
                }),
            },
        )),
    )
}

fn emit_end(handle: &TransportHandle, invocation_id: &str, error: Option<String>) -> Result<()> {
    emit(
        handle,
        &server(run_inference_server_message::Message::InvocationEnd(
            RunInferenceInvocationEnd {
                invocation_id: invocation_id.to_owned(),
                error: error.map(|message| RunInferenceInvocationError { code: 2, message }),
            },
        )),
    )
}

fn server(message: run_inference_server_message::Message) -> RunInferenceServerMessage {
    RunInferenceServerMessage {
        message: Some(message),
    }
}

fn emit(handle: &TransportHandle, message: &RunInferenceServerMessage) -> Result<()> {
    let frame = connect::encode_message(message)?;
    if handle.emit_frame(frame) {
        Ok(())
    } else {
        Err(Error::RunNotFound(handle.request_id().to_owned()))
    }
}
