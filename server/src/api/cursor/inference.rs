//! Serves the Cursor local-loop RunInference stream.
//!
//! The local loop's first append is a RunInferenceClientMessage. Its run
//! request carries a conversation id string, not a ConversationStateStructure.
//! Replies on the open RunInferenceSSE stream are RunInferenceServerMessage
//! frames: run_ready, then invocation text, then invocation_end.

use std::{collections::HashMap, io::Read, sync::OnceLock, time::Duration};

use base64::{engine::general_purpose::STANDARD, Engine};
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
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
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

pub async fn project_prompt(
    request: &InferenceStreamRequest,
    routing_fallback: &str,
) -> Result<ModelRequest> {
    let mut instructions = String::new();
    let mut history = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        let parts = message_parts(message).await?;
        if parts.is_empty() {
            continue;
        }
        match message.role {
            ROLE_SYSTEM => push_system(&mut instructions, &mut history, index, parts),
            ROLE_ASSISTANT if parts_are_text(&parts) => history.push(ProjectedMessage {
                message_id: format!("inference-{index}"),
                role: Role::Assistant,
                content: ProjectedContent::Assistant {
                    text: join_text(&parts),
                    thinking: String::new(),
                    replay_state: None,
                    calls: Vec::new(),
                },
            }),
            ROLE_ASSISTANT => history.push(projected_parts(index, Role::Assistant, parts)),
            _ => history.push(projected_parts(index, Role::User, parts)),
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
        return Err(Error::Protocol(
            "inference request has no user content".into(),
        ));
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

fn push_system(
    instructions: &mut String,
    history: &mut Vec<ProjectedMessage>,
    index: usize,
    parts: Vec<ContentPart>,
) {
    let mut media = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                if !instructions.is_empty() {
                    instructions.push('\n');
                }
                instructions.push_str(&text);
            }
            image => media.push(image),
        }
    }
    if !media.is_empty() {
        history.push(projected_parts(index, Role::User, media));
    }
}

fn projected_parts(index: usize, role: Role, parts: Vec<ContentPart>) -> ProjectedMessage {
    ProjectedMessage {
        message_id: format!("inference-{index}"),
        role,
        content: ProjectedContent::Parts(parts),
    }
}

fn parts_are_text(parts: &[ContentPart]) -> bool {
    parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
}

fn join_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            ContentPart::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn message_parts(message: &InferenceCoreMessage) -> Result<Vec<ContentPart>> {
    match &message.content {
        Some(inference_core_message::Content::Text(text)) => {
            if text.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![ContentPart::Text { text: text.clone() }])
            }
        }
        Some(inference_core_message::Content::Parts(parts)) => {
            let mut output = Vec::new();
            for part in &parts.parts {
                if let Some(part) = content_part(part).await? {
                    output.push(part);
                }
            }
            Ok(output)
        }
        Some(inference_core_message::Content::ToolContent(content)) => {
            let text = content
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
                .join("\n");
            if text.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![ContentPart::Text { text }])
            }
        }
        None => Ok(Vec::new()),
    }
}

async fn content_part(part: &InferenceContentPart) -> Result<Option<ContentPart>> {
    match &part.part {
        Some(inference_content_part::Part::Text(text)) if !text.text.is_empty() => {
            Ok(Some(ContentPart::Text {
                text: text.text.clone(),
            }))
        }
        Some(inference_content_part::Part::Image(image)) => {
            Ok(Some(attached_media(&image.data, &image.mime_type).await?))
        }
        Some(inference_content_part::Part::File(file))
            if visual_hint(&file.media_type, &file.data) =>
        {
            Ok(Some(attached_media(&file.data, &file.media_type).await?))
        }
        Some(inference_content_part::Part::File(file)) => {
            let name = if file.filename.is_empty() {
                "file"
            } else {
                file.filename.as_str()
            };
            Ok(Some(ContentPart::Text {
                text: format!("[File: {name}]"),
            }))
        }
        _ => Ok(None),
    }
}

fn visual_hint(mime: &str, data: &str) -> bool {
    is_visual_mime(mime)
        || data.trim().starts_with("data:image/")
        || data.trim().starts_with("data:video/")
}

pub async fn attached_media(data: &str, mime_hint: &str) -> Result<ContentPart> {
    let data = data.trim();
    if data.is_empty() {
        return Err(Error::Protocol("attached media is empty".into()));
    }
    if let Some(url) = remote_url(data) {
        return fetch_remote_media(url, mime_hint).await;
    }
    let (mime, bytes) = inline_media(data, mime_hint)?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(Error::Protocol(
            "attached media exceeds the size limit".into(),
        ));
    }
    if !is_visual_mime(&mime) {
        return Err(Error::Protocol(format!(
            "attached media type {mime} is not an image or video"
        )));
    }
    Ok(ContentPart::Image {
        mime_type: mime,
        data: bytes,
    })
}

fn remote_url(data: &str) -> Option<&str> {
    if data.starts_with("https://") || data.starts_with("http://") {
        Some(data)
    } else {
        None
    }
}

fn inline_media(data: &str, mime_hint: &str) -> Result<(String, Vec<u8>)> {
    if let Some(rest) = data.strip_prefix("data:") {
        let (header, payload) = rest
            .split_once(',')
            .ok_or_else(|| Error::Protocol("attached media data URL has no payload".into()))?;
        let declared = header.split(';').next().unwrap_or("").trim();
        let bytes = if header.split(';').any(|part| part.trim() == "base64") {
            decode_base64(payload)?
        } else {
            payload.as_bytes().to_vec()
        };
        return Ok((choose_mime(declared, mime_hint, &bytes), bytes));
    }
    let bytes = decode_base64(data)?;
    Ok((choose_mime("", mime_hint, &bytes), bytes))
}

fn choose_mime(declared: &str, hint: &str, bytes: &[u8]) -> String {
    let declared = declared.trim();
    if is_visual_mime(declared) {
        return declared.to_ascii_lowercase();
    }
    let hint = hint.trim();
    if is_visual_mime(hint) {
        return hint.to_ascii_lowercase();
    }
    sniff_mime(bytes)
        .unwrap_or("application/octet-stream")
        .to_string()
}

fn is_visual_mime(mime: &str) -> bool {
    let mime = mime.trim().to_ascii_lowercase();
    mime.starts_with("image/") || mime.starts_with("video/")
}

fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        Some("video/mp4")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("video/webm")
    } else {
        None
    }
}

fn decode_base64(payload: &str) -> Result<Vec<u8>> {
    let compact: String = payload
        .chars()
        .filter(|char| !char.is_whitespace())
        .collect();
    STANDARD
        .decode(compact.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(compact.as_bytes()))
        .map_err(|_| Error::Protocol("attached media is not valid base64".into()))
}

async fn fetch_remote_media(url: &str, mime_hint: &str) -> Result<ContentPart> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(4))
        .build()
        .map_err(|error| Error::Protocol(format!("attached media client: {error}")))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| Error::Protocol("attached media could not be downloaded".into()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(Error::Protocol(format!(
            "attached media download returned {status}"
        )));
    }
    let header_mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .unwrap_or("")
        .trim()
        .to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|_| Error::Protocol("attached media download ended early".into()))?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(Error::Protocol(
            "attached media exceeds the size limit".into(),
        ));
    }
    let mime = choose_mime(&header_mime, mime_hint, &bytes);
    if !is_visual_mime(&mime) {
        return Err(Error::Protocol(format!(
            "attached media type {mime} is not an image or video"
        )));
    }
    Ok(ContentPart::Image {
        mime_type: mime,
        data: bytes.to_vec(),
    })
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
    let mut projected = project_prompt(&request, &routing_text).await?;
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
