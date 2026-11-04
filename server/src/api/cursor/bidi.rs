//! Accepts ordered Cursor Bidi append requests and routes them by request_id.
use prost::Message;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{
            events,
            proto::{agent::v1 as agent, aiserver::v1 as ai},
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

pub fn decode(request: &ai::BidiAppendRequest) -> Result<DecodedAppend> {
    let request_id = request
        .request_id
        .as_ref()
        .map(|id| id.request_id.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::Protocol("BidiAppend request_id is required".into()))?;
    let payload = agent_payload(request)?;
    let message = agent::AgentClientMessage::decode(payload.as_slice()).map_err(|error| {
        Error::Protocol(format!(
            "protobuf decode error: {error}; {}",
            wire_outline(&payload)
        ))
    })?;
    Ok(DecodedAppend {
        request_id: request_id.into(),
        seqno: request.append_seqno,
        message,
    })
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
