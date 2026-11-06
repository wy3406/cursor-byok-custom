//! Accepts ordered Cursor Bidi append requests and routes them by request_id.
use prost::Message;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{
            events,
            proto::{agent::v1 as agent, aiserver::v1 as ai, inference::v1 as inf},
        },
        transport::{TransportParent, TransportRegistry},
    },
    Error, Result,
};

pub struct DecodedAppend {
    pub request_id: String,
    pub seqno: i64,
    pub message: agent::AgentClientMessage,
}

impl DecodedAppend {
    pub fn model_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str())
            .filter(|model| !model.is_empty())
            .or_else(|| {
                request
                    .model_details
                    .as_ref()
                    .map(|model| model.model_id.as_str())
                    .filter(|model| !model.is_empty())
            })
    }

    pub fn conversation_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request.conversation_id.as_deref()
    }

    pub fn is_background_task_completion(&self) -> bool {
        let Some(agent::agent_client_message::Message::RunRequest(request)) =
            self.message.message.as_ref()
        else {
            return false;
        };
        matches!(
            request
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(agent::conversation_action::Action::BackgroundTaskCompletionAction(_))
        )
    }

    pub fn trace_metadata(&self) -> serde_json::Value {
        let Some(message) = self.message.message.as_ref() else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": "empty",
            });
        };
        let agent::agent_client_message::Message::RunRequest(request) = message else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": client_message_type(message),
            });
        };
        let (action_type, history_messages, history_images) = request
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref())
            .map(|action| match action {
                agent::conversation_action::Action::UserMessageAction(action) => {
                    let history = action.conversation_history.as_ref();
                    (
                        "user_message",
                        history.map_or(0, |history| history.messages.len()),
                        history.map_or(0, history_image_count),
                    )
                }
                agent::conversation_action::Action::BackgroundTaskCompletionAction(_) => {
                    ("background_task_completion", 0, 0)
                }
                agent::conversation_action::Action::ExecutePlanAction(_) => ("execute_plan", 0, 0),
                agent::conversation_action::Action::SummarizeAction(_) => ("summarize", 0, 0),
                _ => ("other", 0, 0),
            })
            .unwrap_or(("none", 0, 0));
        let state = request.conversation_state.as_ref();
        serde_json::json!({
            "append_seqno": self.seqno,
            "message_type": "run_request",
            "conversation_id": request.conversation_id,
            "model_id": self.model_id(),
            "action_type": action_type,
            "conversation_history_messages": history_messages,
            "conversation_history_images": history_images,
            "root_message_count": state.map_or(0, |state| state.root_prompt_messages_json.len()),
            "turn_count": state.map_or(0, |state| state.turns.len()),
            "prefetched_blob_count": request.pre_fetched_blobs.len(),
        })
    }
}

fn client_message_type(message: &agent::agent_client_message::Message) -> &'static str {
    use agent::agent_client_message::Message;
    match message {
        Message::RunRequest(_) => "run_request",
        Message::ExecClientMessage(_) => "exec_client_message",
        Message::ExecClientControlMessage(_) => "exec_client_control_message",
        Message::KvClientMessage(_) => "kv_client_message",
        Message::ConversationAction(_) => "conversation_action",
        Message::InteractionResponse(_) => "interaction_response",
        Message::ClientHeartbeat(_) => "client_heartbeat",
        Message::PrewarmRequest(_) => "prewarm_request",
    }
}

fn history_image_count(history: &agent::ConversationHistory) -> usize {
    use agent::{
        conversation_history_message::Message,
        conversation_history_tool_result_content::Content as ToolContent,
        conversation_history_user_content::Content as UserContent,
    };
    history
        .messages
        .iter()
        .map(|message| match message.message.as_ref() {
            Some(Message::User(user)) => user
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(UserContent::Image(_))))
                .count(),
            Some(Message::Tool(tool)) => tool
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(ToolContent::Image(_))))
                .count(),
            _ => 0,
        })
        .sum()
}

#[allow(clippy::large_enum_variant)]
pub enum ClientAppend {
    Agent(DecodedAppend),
    Inference(InferenceAppend),
}

pub struct InferenceAppend {
    pub request_id: String,
    pub seqno: i64,
    pub message: inf::run_inference_client_message::Message,
}

impl ClientAppend {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Agent(append) => append.request_id.as_str(),
            Self::Inference(append) => append.request_id.as_str(),
        }
    }

    pub fn model_id(&self) -> Option<&str> {
        match self {
            Self::Agent(append) => append.model_id(),
            Self::Inference(append) => inference_model_id(&append.message),
        }
    }

    pub fn conversation_id(&self) -> Option<&str> {
        match self {
            Self::Agent(append) => append.conversation_id(),
            Self::Inference(append) => inference_conversation_id(&append.message),
        }
    }

    pub fn trace_metadata(&self) -> serde_json::Value {
        match self {
            Self::Agent(append) => append.trace_metadata(),
            Self::Inference(append) => inference_trace(append),
        }
    }
}

pub fn decode_client(request: &ai::BidiAppendRequest) -> Result<ClientAppend> {
    let request_id = required_request_id(request)?;
    let payload = agent_payload(request)?;
    if let Ok(message) = agent::AgentClientMessage::decode(payload.as_slice()) {
        if agent_owns_append(&message) {
            return Ok(ClientAppend::Agent(DecodedAppend {
                request_id,
                seqno: request.append_seqno,
                message,
            }));
        }
    }
    if let Ok(message) = inf::RunInferenceClientMessage::decode(payload.as_slice()) {
        if let Some(message) = message.message.filter(inference_arm_plausible) {
            return Ok(ClientAppend::Inference(InferenceAppend {
                request_id,
                seqno: request.append_seqno,
                message,
            }));
        }
    }
    decode(request).map(ClientAppend::Agent)
}

fn required_request_id(request: &ai::BidiAppendRequest) -> Result<String> {
    request
        .request_id
        .as_ref()
        .map(|id| id.request_id.clone())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::Protocol("BidiAppend request_id is required".into()))
}

fn inference_arm_plausible(message: &inf::run_inference_client_message::Message) -> bool {
    use inf::run_inference_client_message::Message;
    match message {
        Message::RunRequest(request) => looks_like_uuid(&request.conversation_id),
        Message::InvokeModel(invoke) => !invoke.invocation_id.is_empty(),
        Message::CancelInvocation(cancel) => !cancel.invocation_id.is_empty(),
        Message::FinishRun(_)
        | Message::ClientHeartbeat(_)
        | Message::AgentEvent(_)
        | Message::RunRequestLog(_)
        | Message::ResolveSubagentModels(_)
        | Message::ClientFirstCallTimings(_) => true,
    }
}

fn looks_like_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    bytes.iter().enumerate().all(|(index, byte)| {
        if matches!(index, 8 | 13 | 18 | 23) {
            *byte == b'-'
        } else {
            byte.is_ascii_hexdigit()
        }
    })
}

fn inference_model_id(message: &inf::run_inference_client_message::Message) -> Option<&str> {
    use inf::run_inference_client_message::Message;
    let model_id = match message {
        Message::RunRequest(request) => request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str()),
        Message::InvokeModel(invoke) => {
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

fn inference_conversation_id(message: &inf::run_inference_client_message::Message) -> Option<&str> {
    use inf::run_inference_client_message::Message;
    let conversation_id = match message {
        Message::RunRequest(request) => request.conversation_id.as_str(),
        Message::InvokeModel(invoke) => invoke
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

fn inference_trace(append: &InferenceAppend) -> serde_json::Value {
    use inf::run_inference_client_message::Message;
    match &append.message {
        Message::RunRequest(request) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_run_request",
            "conversation_id": request.conversation_id,
            "model_id": inference_model_id(&append.message),
            "routing_messages": request.routing_conversation.len(),
        }),
        Message::InvokeModel(invoke) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_invoke_model",
            "invocation_id": invoke.invocation_id,
            "message_count": invoke.request.as_ref().map(|request| request.messages.len()).unwrap_or(0),
            "compressed": invoke.compressed_request.is_some(),
        }),
        Message::CancelInvocation(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_cancel_invocation",
        }),
        Message::FinishRun(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_finish_run",
        }),
        Message::ClientHeartbeat(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_client_heartbeat",
        }),
        Message::AgentEvent(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_agent_event",
        }),
        Message::RunRequestLog(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_run_request_log",
        }),
        Message::ResolveSubagentModels(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_resolve_subagent_models",
        }),
        Message::ClientFirstCallTimings(_) => serde_json::json!({
            "append_seqno": append.seqno,
            "message_type": "inference_client_first_call_timings",
        }),
    }
}

pub fn decode(request: &ai::BidiAppendRequest) -> Result<DecodedAppend> {
    let request_id = required_request_id(request)?;
    let payload = agent_payload(request)?;
    let message = agent::AgentClientMessage::decode(payload.as_slice()).map_err(|error| {
        Error::Protocol(format!(
            "protobuf decode error: {error}; {}",
            wire_outline(&payload)
        ))
    })?;
    if bare_interaction_response(&message) {
        return Err(Error::Protocol(format!(
            "protobuf decode error: AgentClientMessage.message field 6 has no interaction id; {}",
            wire_outline(&payload)
        )));
    }
    Ok(DecodedAppend {
        request_id,
        seqno: request.append_seqno,
        message,
    })
}

fn agent_owns_append(message: &agent::AgentClientMessage) -> bool {
    !bare_interaction_response(message) && message.message.is_some()
}

fn bare_interaction_response(message: &agent::AgentClientMessage) -> bool {
    matches!(
        &message.message,
        Some(agent::agent_client_message::Message::InteractionResponse(response))
            if response.id == 0 && response.result.is_none()
    )
}

fn agent_payload(request: &ai::BidiAppendRequest) -> Result<Vec<u8>> {
    if !request.data_binary.is_empty() {
        return Ok(request.data_binary.clone());
    }
    let data = request.data.trim();
    if data.is_empty() {
        return Err(Error::Protocol(
            "BidiAppend contains no AgentClientMessage".into(),
        ));
    }
    hex::decode(data).map_err(|error| {
        let position = match error {
            hex::FromHexError::InvalidHexCharacter { index, .. } => index.to_string(),
            hex::FromHexError::OddLength => "end".to_string(),
            hex::FromHexError::InvalidStringLength => "length".to_string(),
        };
        Error::Protocol(format!("invalid BidiAppend hex at position {position}"))
    })
}

const WIRE_OUTLINE_TAGS: usize = 40;
const WIRE_OUTLINE_DEPTH: usize = 4;

fn wire_outline(bytes: &[u8]) -> String {
    let mut tags = String::new();
    let mut count = 0;
    walk_wire(bytes, 0, 0, &mut tags, &mut count);
    if count >= WIRE_OUTLINE_TAGS {
        tags.push_str(" truncated");
    }
    format!("wire len={}{tags}", bytes.len())
}

fn walk_wire(bytes: &[u8], depth: usize, base: usize, tags: &mut String, count: &mut usize) {
    let mut offset = 0;
    while offset < bytes.len() && *count < WIRE_OUTLINE_TAGS {
        let start = offset;
        let Some((key, next)) = read_varint(bytes, offset) else {
            push_tag(tags, depth, base + start, 0, "truncated-key", None);
            *count += 1;
            return;
        };
        offset = next;
        if key > u64::from(u32::MAX) {
            push_tag(tags, depth, base + start, 0, "invalid-key", None);
            *count += 1;
            return;
        }
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        if field == 0 {
            push_tag(tags, depth, base + start, 0, "invalid-field", None);
            *count += 1;
            return;
        }
        match wire {
            0 => {
                let Some((_, next)) = read_varint(bytes, offset) else {
                    push_tag(tags, depth, base + start, field, "v", None);
                    *count += 1;
                    return;
                };
                push_tag(tags, depth, base + start, field, "v", None);
                *count += 1;
                offset = next;
            }
            1 => {
                push_tag(tags, depth, base + start, field, "i64", None);
                *count += 1;
                if bytes.len() - offset < 8 {
                    return;
                }
                offset += 8;
            }
            5 => {
                push_tag(tags, depth, base + start, field, "i32", None);
                *count += 1;
                if bytes.len() - offset < 4 {
                    return;
                }
                offset += 4;
            }
            2 => {
                let Some((len, next)) = read_varint(bytes, offset) else {
                    push_tag(tags, depth, base + start, field, "ld", None);
                    *count += 1;
                    return;
                };
                offset = next;
                let len = usize::try_from(len).unwrap_or(usize::MAX);
                push_tag(tags, depth, base + start, field, "ld", Some(len));
                *count += 1;
                if len > bytes.len() - offset {
                    return;
                }
                if depth + 1 < WIRE_OUTLINE_DEPTH {
                    walk_wire(
                        &bytes[offset..offset + len],
                        depth + 1,
                        base + offset,
                        tags,
                        count,
                    );
                }
                offset += len;
            }
            3 => {
                push_tag(tags, depth, base + start, field, "sg", None);
                *count += 1;
            }
            4 => {
                push_tag(tags, depth, base + start, field, "eg", None);
                *count += 1;
                return;
            }
            other => {
                push_tag(tags, depth, base + start, field, "w", Some(other as usize));
                *count += 1;
                return;
            }
        }
    }
}

fn push_tag(
    tags: &mut String,
    depth: usize,
    offset: usize,
    field: u32,
    wire: &str,
    extra: Option<usize>,
) {
    tags.push(' ');
    tags.push_str(&format!("d{depth}@{offset}#{field}/{wire}"));
    if let Some(extra) = extra {
        tags.push(':');
        tags.push_str(&extra.to_string());
    }
}

fn read_varint(bytes: &[u8], mut offset: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    while offset < bytes.len() && shift < 64 {
        let byte = bytes[offset];
        offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, offset));
        }
        shift += 7;
    }
    None
}

pub async fn append(
    registry: &TransportRegistry,
    request: DecodedAppend,
    parent: Option<TransportParent>,
) -> Result<ai::BidiAppendResponse> {
    let replace_closing = request.model_id().is_some();
    let handle = registry
        .get_or_create_for_append(&request.request_id, replace_closing)
        .await?;
    let _admission = handle.admit()?;
    if let Some(conversation_id) = request.conversation_id() {
        handle.set_conversation_id(conversation_id)?;
    }
    if let Some(parent) = parent {
        handle.set_parent(parent)?;
    }
    if matches!(
        request.message.message.as_ref(),
        Some(agent::agent_client_message::Message::ClientHeartbeat(_))
    ) {
        handle.emit(&events::heartbeat())?;
    }
    handle
        .command(TransportCommand::Append {
            seqno: request.seqno,
            message: Box::new(request.message),
        })
        .await?;
    Ok(ai::BidiAppendResponse {})
}
