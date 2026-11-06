//! Serves the Cursor local-loop RunInference stream.
//!
//! The local loop's first append is a RunInferenceClientMessage. Its run
//! request carries a conversation id string, not a ConversationStateStructure.
//! Replies on the open RunInferenceSSE stream are RunInferenceServerMessage
//! frames: run_ready, then invocation text, then invocation_end.

use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    sync::OnceLock,
    time::Duration,
};

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
                run_inference_client_message, run_inference_server_message, InferenceAgentTool,
                InferenceContentPart, InferenceCoreMessage, InferenceStreamRequest,
                InferenceStreamResponse, InferenceTextStreamPart, InferenceThinkingStreamPart,
                InferenceToolCall, InferenceToolCallStreamPart, InferenceToolResultContent,
                RunInferenceCompressedRequest, RunInferenceHeartbeat, RunInferenceInvocationEnd,
                RunInferenceInvocationError, RunInferenceInvocationResponse,
                RunInferenceInvokeModel, RunInferencePromptModelMetadata,
                RunInferenceRoutingMessage, RunInferenceRunReady, RunInferenceRunRequest,
                RunInferenceServerMessage,
            },
        },
        transport::{TransportHandle, TransportParent, TransportRegistry},
    },
    model::{
        ContentPart, ModelInvocation, ModelRequest, ModelSpec, ModelType, ProjectedContent,
        ProjectedMessage, PromptSpec, Role, ToolCallContent, ToolDefinition, ToolResultContent,
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
            open_run(registry, &handle, request).await
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
        run_inference_client_message::Message::AgentEvent(_)
        | run_inference_client_message::Message::RunRequestLog(_)
        | run_inference_client_message::Message::ResolveSubagentModels(_)
        | run_inference_client_message::Message::ClientFirstCallTimings(_) => Ok(()),
    }
}

pub async fn project_prompt(
    request: &InferenceStreamRequest,
    routing_fallback: &str,
) -> Result<ModelRequest> {
    let mut instructions = String::new();
    let mut history = Vec::new();
    for (index, message) in request.messages.iter().enumerate() {
        if let Some(inference_core_message::Content::ToolContent(content)) = &message.content {
            push_tool_results(&mut history, index, content);
            continue;
        }
        if message.role == ROLE_ASSISTANT && !message.tool_calls.is_empty() {
            let parts = message_parts(message).await?;
            history.push(assistant_with_calls(index, &parts, &message.tool_calls));
            continue;
        }
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
            tools: project_tools(&request.tools),
        },
        model: ModelSpec::new(model_id),
        history,
    })
}

fn project_tools(tools: &[InferenceAgentTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .filter(|tool| !tool.name.is_empty())
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: restore_empty_tool_schema(&tool.name, parameters_value(&tool.parameters)),
        })
        .collect()
}

fn push_tool_results(
    history: &mut Vec<ProjectedMessage>,
    index: usize,
    content: &InferenceToolResultContent,
) {
    for (part_index, part) in content.parts.iter().enumerate() {
        history.push(ProjectedMessage {
            message_id: format!("inference-{index}-{part_index}"),
            role: Role::Tool,
            content: ProjectedContent::ToolResult(ToolResultContent {
                call_id: part.tool_call_id.clone(),
                name: part.tool_name.clone(),
                content: value_text(&part.result),
                is_error: part.is_error,
                image: None,
                provider_parts: Vec::new(),
            }),
        });
    }
}

fn assistant_with_calls(
    index: usize,
    parts: &[ContentPart],
    tool_calls: &[InferenceToolCall],
) -> ProjectedMessage {
    ProjectedMessage {
        message_id: format!("inference-{index}"),
        role: Role::Assistant,
        content: ProjectedContent::Assistant {
            text: join_text(parts),
            thinking: String::new(),
            replay_state: None,
            calls: tool_calls
                .iter()
                .enumerate()
                .map(|(call_index, call)| ToolCallContent {
                    index: call_index,
                    call_id: call.tool_call_id.clone(),
                    name: call.tool_name.clone(),
                    arguments: tool_call_arguments(call),
                })
                .collect(),
        },
    }
}

fn tool_call_arguments(call: &InferenceToolCall) -> serde_json::Value {
    if let Ok(value) = serde_json::from_str(&call.raw_tool_call_args) {
        return value;
    }
    parameters_value(&call.args)
}

fn parameters_value(bytes: &[u8]) -> serde_json::Value {
    if bytes.is_empty() {
        return serde_json::json!({});
    }
    if let Some(value) = json_object_bytes(bytes) {
        return unwrap_embedded_schema(value);
    }
    if let Ok(message) = prost_types::Struct::decode(bytes) {
        let value = unwrap_embedded_schema(struct_json(&message));
        if json_schema_has_fields(&value)
            || value.as_object().is_some_and(|object| !object.is_empty())
        {
            return value;
        }
    }
    if let Ok(message) = prost_types::Value::decode(bytes) {
        if let Some(Kind::StringValue(text)) = message.kind.as_ref() {
            if let Some(value) = json_object_bytes(text.as_bytes()) {
                return unwrap_embedded_schema(value);
            }
        }
    }
    serde_json::json!({})
}

fn json_object_bytes(bytes: &[u8]) -> Option<serde_json::Value> {
    let value = serde_json::from_slice::<serde_json::Value>(bytes).ok()?;
    match value {
        serde_json::Value::Object(_) => Some(value),
        serde_json::Value::String(text) => {
            let nested = serde_json::from_str::<serde_json::Value>(&text).ok()?;
            nested.is_object().then_some(nested)
        }
        _ => None,
    }
}

fn unwrap_embedded_schema(value: serde_json::Value) -> serde_json::Value {
    if json_schema_has_fields(&value) {
        return value;
    }
    let Some(object) = value.as_object() else {
        return value;
    };
    for key in ["jsonSchema", "input_schema", "parameters", "schema"] {
        if let Some(inner) = object.get(key).filter(|inner| inner.is_object()).cloned() {
            return unwrap_embedded_schema(inner);
        }
    }
    value
}

fn restore_empty_tool_schema(name: &str, parameters: serde_json::Value) -> serde_json::Value {
    if !parameters
        .as_object()
        .is_some_and(|object| object.is_empty())
    {
        return parameters;
    }
    known_latest_tool_schema(name).unwrap_or(parameters)
}

fn known_latest_tool_schema(name: &str) -> Option<serde_json::Value> {
    let string =
        |description: &str| serde_json::json!({ "type": "string", "description": description });
    let integer =
        |description: &str| serde_json::json!({ "type": "integer", "description": description });
    let schema = match name {
        "Read" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": string("The absolute path of the file to read."),
                "offset": integer("The line number to start reading from."),
                "limit": integer("The number of lines to read.")
            },
            "required": ["path"]
        }),
        "Delete" => serde_json::json!({
            "type": "object",
            "properties": {
                "path": string("The absolute path of the file to delete.")
            },
            "required": ["path"]
        }),
        "Shell" => serde_json::json!({
            "type": "object",
            "properties": {
                "command": string("The command to execute."),
                "working_directory": string("The absolute path of the working directory.")
            },
            "required": ["command"]
        }),
        "Grep" => serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": string("The regular expression pattern to search for in file contents."),
                "path": string("File or directory to search in. Defaults to the workspace root."),
                "glob": string("Glob pattern to filter files."),
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode."
                }
            },
            "required": ["pattern"]
        }),
        "Glob" => serde_json::json!({
            "type": "object",
            "properties": {
                "glob_pattern": string("The glob pattern to match files against."),
                "target_directory": string("Absolute path of the directory to search. Defaults to the workspace root.")
            },
            "required": ["glob_pattern"]
        }),
        "LS" => serde_json::json!({
            "type": "object",
            "properties": {
                "target_directory": string("Absolute path of the directory to list.")
            },
            "required": ["target_directory"]
        }),
        "GetMcpTools" => serde_json::json!({
            "type": "object",
            "properties": {
                "server": string("MCP server identifier to inspect."),
                "toolName": string("Tool name within the server. Requires server to be set."),
                "pattern": string("RE2 pattern to search server and tool names.")
            }
        }),
        _ => return None,
    };
    Some(schema)
}

fn json_schema_has_fields(value: &serde_json::Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    ["properties", "anyOf", "oneOf", "allOf", "items"]
        .into_iter()
        .any(|key| object.contains_key(key))
}

fn struct_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_json(value)))
            .collect(),
    )
}

fn prost_json(value: &prost_types::Value) -> serde_json::Value {
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::NumberValue(number)) => serde_json::Number::from_f64(*number)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(text)) => serde_json::Value::String(text.clone()),
        Some(Kind::BoolValue(flag)) => serde_json::Value::Bool(*flag),
        Some(Kind::StructValue(nested)) => struct_json(nested),
        Some(Kind::ListValue(values)) => {
            serde_json::Value::Array(values.values.iter().map(prost_json).collect())
        }
    }
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
    if bytes.is_empty() {
        return String::new();
    }
    if let Ok(value) = prost_types::Value::decode(bytes) {
        return kind_text(value.kind.as_ref(), 0);
    }
    if let Some(text) = struct_text(bytes) {
        return text;
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    String::new()
}

fn struct_text(bytes: &[u8]) -> Option<String> {
    let message = prost_types::Struct::decode(bytes).ok()?;
    let text = serde_json::to_string(&struct_json(&message)).ok()?;
    (text != "{}").then_some(text)
}

fn kind_text(kind: Option<&Kind>, depth: usize) -> String {
    if depth > 12 {
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

pub fn local_prompt_metadata(
    model_type: Option<ModelType>,
    display_name: &str,
    provider_model_id: &str,
) -> RunInferencePromptModelMetadata {
    let name = format!("{display_name} {provider_model_id}").to_ascii_lowercase();
    let mut metadata = RunInferencePromptModelMetadata {
        prompt_version: "latest".to_owned(),
        self_identity_name: prompt_self_identity(display_name, provider_model_id),
        ..Default::default()
    };

    if name.contains("gemini") {
        metadata.vendor = "google".to_owned();
        metadata.is_gemini3 = true;
    } else if name.contains("grok") {
        metadata.vendor = "xai".to_owned();
        if contains_any(&name, &["4.7", "4-7", "4.6", "4-6"]) {
            metadata.is_grok46_product_prompt = true;
        } else if contains_any(&name, &["4.5", "4-5"]) {
            metadata.is_grok45_product_prompt = true;
        } else {
            metadata.is_grok46_product_prompt = true;
        }
    } else if name.contains("haiku") {
        metadata.vendor = "anthropic".to_owned();
        metadata.prompt_version = "haiku".to_owned();
        metadata.is_claude_4x = true;
    } else if name.contains("fable") {
        metadata.vendor = "anthropic".to_owned();
        metadata.is_claude_4x = true;
        if contains_any(&name, &["5.1", "5-1"]) {
            metadata.is_fable51 = true;
        } else {
            metadata.is_fable5 = true;
        }
    } else if name.contains("opus") {
        metadata.vendor = "anthropic".to_owned();
        metadata.is_claude_4x = true;
        if contains_any(&name, &["5.5", "5-5"]) {
            metadata.is_opus55 = true;
        } else if contains_any(&name, &["4.8", "4-8"]) {
            metadata.is_opus48 = true;
        } else if contains_any(&name, &["4.6", "4-6"]) {
            metadata.is_opus46 = true;
        } else if contains_any(&name, &["4.5", "4-5"]) {
            metadata.is_opus45 = true;
        } else {
            metadata.is_opus5 = true;
        }
    } else if name.contains("sonnet") {
        metadata.vendor = "anthropic".to_owned();
        metadata.is_claude_4x = true;
        if contains_any(&name, &["4.5", "4-5"]) {
            metadata.is_sonnet45 = true;
        } else {
            metadata.is_sonnet4 = true;
        }
    } else if name.contains("claude") {
        metadata.vendor = "anthropic".to_owned();
        metadata.is_claude_4x = true;
    } else if name.contains("matterhorn") {
        metadata.vendor = "cursor".to_owned();
        metadata.is_composer_matterhorn = true;
        metadata.is_composer2 = true;
    } else if name.contains("composer") {
        metadata.vendor = "cursor".to_owned();
        if contains_any(&name, &["1.5", "1-5"]) {
            metadata.is_composer15 = true;
        } else if contains_any(&name, &["composer-2", "composer2", "composer 2"]) {
            metadata.is_composer2 = true;
        } else {
            metadata.is_composer1 = true;
        }
    } else if name.contains("glm") || name.contains("zai") {
        metadata.vendor = "zai".to_owned();
    } else if name.contains("codex") {
        metadata.vendor = "openai".to_owned();
        metadata.prompt_version = "gpt5-codex".to_owned();
        metadata.is_codex_family = true;
        metadata.is_gpt5_family = true;
        if name.contains("spark") {
            metadata.is_gpt53_codex_spark = true;
        } else if contains_any(&name, &["5.3", "5-3"]) {
            metadata.is_gpt53_codex = true;
        } else if contains_any(&name, &["5.2", "5-2"]) {
            metadata.is_gpt52_codex = true;
        }
    } else if contains_any(&name, &["gpt-6", "gpt6", "gpt-5.6", "gpt-5-6", "gpt56"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt56 = true;
        metadata.is_gpt5_family = true;
    } else if contains_any(&name, &["gpt-5.5", "gpt-5-5"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt55 = true;
        metadata.is_gpt5_family = true;
    } else if contains_any(&name, &["gpt-5.4", "gpt-5-4"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt54 = true;
        metadata.is_gpt5_family = true;
    } else if contains_any(&name, &["gpt-5.2", "gpt-5-2"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt52 = true;
        metadata.is_gpt5_family = true;
    } else if contains_any(&name, &["gpt-5.1", "gpt-5-1"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt51 = true;
        metadata.is_gpt5_family = true;
    } else if contains_any(&name, &["gpt-5", "gpt5", "gpt"]) {
        metadata.vendor = "openai".to_owned();
        metadata.is_gpt5 = true;
        metadata.is_gpt5_family = true;
    } else {
        metadata.vendor = match model_type {
            Some(ModelType::OpenAi) => "openai",
            Some(ModelType::Anthropic) => "anthropic",
            None => "unknown",
        }
        .to_owned();
    }

    metadata
}

fn prompt_self_identity(display_name: &str, provider_model_id: &str) -> String {
    if !display_name.is_empty() {
        display_name.to_owned()
    } else if !provider_model_id.is_empty() && !is_sixteen_hex(provider_model_id) {
        provider_model_id.to_owned()
    } else {
        String::new()
    }
}

fn is_sixteen_hex(value: &str) -> bool {
    value.len() == 16 && value.chars().all(|character| character.is_ascii_hexdigit())
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

async fn open_run(
    registry: &TransportRegistry,
    handle: &TransportHandle,
    request: RunInferenceRunRequest,
) -> Result<()> {
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
    let model_id = model.model_id.clone();
    let metadata = if model_id.is_empty() {
        local_prompt_metadata(None, "", "")
    } else {
        match registry.store().model(&model_id).await? {
            Some(stored) => local_prompt_metadata(
                Some(stored.model_type),
                &stored.display_name,
                &stored.model_id,
            ),
            None => local_prompt_metadata(None, "", &model_id),
        }
    };
    emit(
        handle,
        &server(run_inference_server_message::Message::RunReady(
            RunInferenceRunReady {
                resolved_model: Some(model),
                prompt_model_metadata: Some(metadata.clone()),
                accepts_client_heartbeat: true,
            },
        )),
    )?;
    spawn_heartbeats(handle.clone(), heartbeats);
    tracing::info!(
        request_id = handle.request_id(),
        model_id,
        prompt_vendor = metadata.vendor.as_str(),
        prompt_version = metadata.prompt_version.as_str(),
        "inference run ready"
    );
    Ok(())
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
    let mut saw_tool = false;
    let mut saw_thinking = false;
    let mut thinking_closed = false;
    let mut open_tools = BTreeMap::<usize, OpenToolCall>::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                close_thinking(handle, invocation_id, saw_thinking, &mut thinking_closed)?;
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
                        saw_thinking = true;
                        emit_thinking(handle, invocation_id, &text, false)?;
                    }
                    ModelEvent::ThinkingEnd => {
                        close_thinking(handle, invocation_id, saw_thinking, &mut thinking_closed)?;
                    }
                    ModelEvent::ToolCallStart { index, call_id, name } => {
                        let call = open_tools.entry(index).or_default();
                        call.call_id = call_id;
                        call.name = name;
                    }
                    ModelEvent::ToolCallArgumentsDelta { index, delta } => {
                        open_tools.entry(index).or_default().arguments.push_str(&delta);
                    }
                    ModelEvent::ToolCallEnd { index } => {
                        let call = open_tools.remove(&index).unwrap_or_default();
                        emit_tool_call(handle, invocation_id, index, &call)?;
                        saw_tool = true;
                    }
                    ModelEvent::Done(_) => break,
                    _ => {}
                }
            }
        }
    }
    for (index, call) in open_tools {
        emit_tool_call(handle, invocation_id, index, &call)?;
        saw_tool = true;
    }
    close_thinking(handle, invocation_id, saw_thinking, &mut thinking_closed)?;
    if !saw_text && !saw_tool && !saw_thinking {
        return Err(Error::Provider(
            "inference provider returned no text".into(),
        ));
    }
    if saw_text {
        emit_text(handle, invocation_id, "", true)?;
    }
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
        run_inference_client_message::Message::CancelInvocation(_)
        | run_inference_client_message::Message::FinishRun(_)
        | run_inference_client_message::Message::ClientHeartbeat(_)
        | run_inference_client_message::Message::AgentEvent(_)
        | run_inference_client_message::Message::RunRequestLog(_)
        | run_inference_client_message::Message::ResolveSubagentModels(_)
        | run_inference_client_message::Message::ClientFirstCallTimings(_) => None,
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
        run_inference_client_message::Message::CancelInvocation(_)
        | run_inference_client_message::Message::FinishRun(_)
        | run_inference_client_message::Message::ClientHeartbeat(_)
        | run_inference_client_message::Message::AgentEvent(_)
        | run_inference_client_message::Message::RunRequestLog(_)
        | run_inference_client_message::Message::ResolveSubagentModels(_)
        | run_inference_client_message::Message::ClientFirstCallTimings(_) => "",
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
        run_inference_client_message::Message::AgentEvent(_) => "agent_event",
        run_inference_client_message::Message::RunRequestLog(_) => "run_request_log",
        run_inference_client_message::Message::ResolveSubagentModels(_) => {
            "resolve_subagent_models"
        }
        run_inference_client_message::Message::ClientFirstCallTimings(_) => {
            "client_first_call_timings"
        }
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

#[derive(Default)]
struct OpenToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

fn emit_tool_call(
    handle: &TransportHandle,
    invocation_id: &str,
    index: usize,
    call: &OpenToolCall,
) -> Result<()> {
    emit(
        handle,
        &server(run_inference_server_message::Message::InvocationResponse(
            RunInferenceInvocationResponse {
                invocation_id: invocation_id.to_owned(),
                response: Some(InferenceStreamResponse {
                    response: Some(inference_stream_response::Response::ToolCallPart(
                        InferenceToolCallStreamPart {
                            tool_call_id: call.call_id.clone(),
                            tool_name: call.name.clone(),
                            args: call.arguments.clone(),
                            is_complete: true,
                            tool_index: i32::try_from(index).unwrap_or(i32::MAX),
                        },
                    )),
                }),
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

fn close_thinking(
    handle: &TransportHandle,
    invocation_id: &str,
    saw_thinking: bool,
    thinking_closed: &mut bool,
) -> Result<()> {
    if saw_thinking && !*thinking_closed {
        emit_thinking(handle, invocation_id, "", true)?;
        *thinking_closed = true;
    }
    Ok(())
}

fn emit_thinking(
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
                    response: Some(inference_stream_response::Response::ThinkingPart(
                        InferenceThinkingStreamPart {
                            text: text.to_owned(),
                            is_final,
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

#[cfg(test)]
mod tests {
    use prost::Message;
    use prost_types::value::Kind;

    fn encode(message: &impl Message) -> Vec<u8> {
        let mut bytes = Vec::new();
        message.encode(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn value_text_keeps_protobuf_string() {
        let value = prost_types::Value {
            kind: Some(Kind::StringValue("Read path".into())),
        };
        assert_eq!(super::value_text(&encode(&value)), "Read path");
    }

    #[test]
    fn value_text_keeps_empty_value_empty() {
        let value = prost_types::Value {
            kind: Some(Kind::NullValue(0)),
        };
        assert!(super::value_text(&encode(&value)).is_empty());
    }

    #[test]
    fn value_text_reads_struct_fields_as_json() {
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "name".into(),
            prost_types::Value {
                kind: Some(Kind::StringValue("Read".into())),
            },
        );
        let message = prost_types::Struct { fields };
        let bytes = encode(&message);
        let text = super::value_text(&bytes);
        assert!(text.contains("\"name\":\"Read\""), "{text}");
        assert_ne!(text, String::from_utf8_lossy(&bytes).trim());
    }

    #[test]
    fn value_text_keeps_plain_utf8_when_protobuf_does_not_decode() {
        let bytes = b"available tools: Read, Glob, Grep";
        assert_eq!(
            super::value_text(bytes),
            "available tools: Read, Glob, Grep"
        );
    }
}
