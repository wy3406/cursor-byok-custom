//! Implements the OpenAI Responses provider adapter.
use async_stream::try_stream;
use base64::{engine::general_purpose::STANDARD, Engine};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

use crate::{
    config::ProviderConfig,
    model::{
        ContentPart, ModelInvocation, ModelLatency, ProjectedContent, ProjectedMessage, Role, Usage,
    },
    Error, Result,
};

use super::{
    apply_body_allowlist, apply_openai_prompt_cache_key,
    attempt::{send_once, Attempt},
    map_sse_error, merge_extra_params, provider_event_error,
    recorder::recorded_headers,
    CallRecorder, FinishReason, ModelEvent, Provider, ProviderStream,
};

#[derive(Default)]
struct ResponseToolState {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    announced_arguments: Option<String>,
    emitted_arguments: usize,
    started: bool,
    ended: bool,
}

enum ResponseToolArguments<'a> {
    None,
    Delta(&'a str),
    Snapshot(&'a str),
}

pub struct OpenAiResponsesProvider {
    client: reqwest::Client,
    config: ProviderConfig,
    recorder: Option<CallRecorder>,
}

impl OpenAiResponsesProvider {
    pub fn new(client: reqwest::Client, config: ProviderConfig) -> Self {
        Self {
            client,
            config,
            recorder: None,
        }
    }

    pub fn with_recorder(mut self, recorder: Option<CallRecorder>) -> Self {
        self.recorder = recorder;
        self
    }
}

impl Provider for OpenAiResponsesProvider {
    fn stream(
        &self,
        invocation: ModelInvocation,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> ProviderStream {
        let client = self.client.clone();
        let config = self.config.clone();
        let recorder = self.recorder.clone();
        Box::pin(try_stream! {
            let ModelInvocation { call_id, request, .. } = invocation;
            let input = responses_input(&request.history)?;
            let mut body = json!({
                "model": request.model.model_id, "input": input, "stream": true,
                "instructions": request.prompt.instructions
            });
            if !request.prompt.tools.is_empty() {
                body["tools"] = json!(request.prompt.tools.iter().map(|tool| json!({
                    "type":"function", "name":tool.name, "description":tool.description,
                    "parameters":tool.parameters, "strict":false
                })).collect::<Vec<_>>());
            }
            apply_model(&mut body, &request.model, config.max_output_tokens)?;
            merge_extra_params(&mut body, &request.model.extra_params)?;
            apply_openai_prompt_cache_key(&mut body, &request.model.model_id)?;
            apply_body_allowlist(&mut body, config.allowed_body_fields.as_ref())?;
            let request_headers = recorded_headers(&config, &[("content-type", "application/json")]);
            if let Some(recorder) = &recorder {
                recorder.request(request_headers.clone(), &body).await?;
            }
            let attempt = send_once(
                "OpenAI Responses",
                || client.post(&config.request_url)
                    .bearer_auth(&config.api_key).headers(config.custom_headers.clone()).json(&body),
                &cancellation,
                recorder.as_ref(),
            ).await?;
            let Attempt::Response(response) = attempt else { return };
            yield ModelEvent::Start { model_call_id: call_id };
            let chunk_recorder = recorder.clone();
            let chunks = response.bytes_stream()
                .map(|chunk| chunk.map_err(Error::from))
                .then(move |chunk| {
                    let recorder = chunk_recorder.clone();
                    async move {
                        let chunk = chunk?;
                        if let Some(recorder) = recorder { recorder.response_chunk(&chunk).await?; }
                        Ok::<_, Error>(chunk)
                    }
                });
            let source = chunks.eventsource();
            futures_util::pin_mut!(source);
            let mut text_open = false;
            let mut text = String::new();
            let mut thinking_open = false;
            let mut reasoning_delta_indexes = std::collections::BTreeSet::<usize>::new();
            let mut unindexed_reasoning_delta = false;
            let mut tools = std::collections::BTreeMap::<usize, ResponseToolState>::new();
            let mut reasoning_items = Vec::new();
            let mut saw_tool = false;
            let mut saw_completed_item = false;
            let mut terminal = false;
            let mut reply_split = ReplySplit::default();
            loop {
                let event = tokio::select! {
                    _ = cancellation.cancelled() => { return; }
                    event = source.next() => event,
                };
                let Some(event) = event else { break };
                let event = event.map_err(|error| map_sse_error("OpenAI Responses", error))?;
                if event.data == "[DONE]" { break; }
                let value: Value = serde_json::from_str(&event.data)?;
                if let Some(error) = provider_event_error("OpenAI Responses", &value) {
                    Err(error)?;
                }
                let kind = value.get("type").and_then(Value::as_str).unwrap_or(&event.event);
                match kind {
                    "response.output_text.delta" => {
                        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                            for event in reply_events(reply_split.push(delta), &mut text_open, &mut text, &mut thinking_open) {
                                yield event;
                            }
                        }
                    }
                    "response.output_text.done" => {
                        if let Some(final_text) = value.get("text").and_then(Value::as_str) {
                            for event in accept_output_text(&mut reply_split, &mut text_open, &mut text, &mut thinking_open, final_text) {
                                yield event;
                            }
                        }
                        for event in reply_events(reply_split.finish(), &mut text_open, &mut text, &mut thinking_open) {
                            yield event;
                        }
                        if text_open { text_open = false; yield ModelEvent::TextEnd; }
                    }
                    "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                        if !thinking_open { thinking_open = true; yield ModelEvent::ThinkingStart; }
                        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                            note_reasoning_delta(
                                &value,
                                &mut reasoning_delta_indexes,
                                &mut unindexed_reasoning_delta,
                            );
                            yield ModelEvent::ThinkingDelta(delta.into());
                        }
                    }
                    "response.reasoning_summary_text.done" | "response.reasoning_text.done"
                        if thinking_open => { thinking_open = false; yield ModelEvent::ThinkingEnd; }
                    "response.output_item.added" => {
                        let item = value.get("item").unwrap_or(&Value::Null);
                        if item.get("type").and_then(Value::as_str) == Some("function_call") {
                            for event in reply_events(reply_split.close_for_tool(), &mut text_open, &mut text, &mut thinking_open) {
                                yield event;
                            }
                            if thinking_open { thinking_open = false; yield ModelEvent::ThinkingEnd; }
                            let index = required_u64(&value, "output_index")? as usize;
                            saw_tool = true;
                            for event in update_response_tool(index, item, ResponseToolArguments::None, false, &mut tools)? { yield event; }
                        }
                    }
                    "response.output_item.done" => {
                        let item = value.get("item").unwrap_or(&Value::Null);
                        match item.get("type").and_then(Value::as_str) {
                            Some("reasoning") => {
                                let streamed = take_reasoning_item_streamed(
                                    &value,
                                    &mut reasoning_delta_indexes,
                                    &mut unindexed_reasoning_delta,
                                );
                                if !thinking_open && !streamed {
                                    let text = reasoning_visible_text(item);
                                    if !text.is_empty() {
                                        yield ModelEvent::ThinkingStart;
                                        yield ModelEvent::ThinkingDelta(text);
                                        yield ModelEvent::ThinkingEnd;
                                    }
                                } else if thinking_open {
                                    thinking_open = false;
                                    yield ModelEvent::ThinkingEnd;
                                }
                                reasoning_items.push(item.clone());
                            }
                            Some("message") => {
                                saw_completed_item = true;
                                let thought = reasoning_visible_text(item);
                                if !thinking_open && !thought.is_empty() {
                                    thinking_open = true;
                                    yield ModelEvent::ThinkingStart;
                                    yield ModelEvent::ThinkingDelta(thought);
                                    reasoning_items.push(json!({
                                        "type": "reasoning",
                                        "summary": reasoning_visible_text(item)
                                    }));
                                }
                                if let Some(final_text) = response_item_text(item) {
                                    for event in accept_output_text(&mut reply_split, &mut text_open, &mut text, &mut thinking_open, &final_text) {
                                        yield event;
                                    }
                                    for event in reply_events(reply_split.finish(), &mut text_open, &mut text, &mut thinking_open) {
                                        yield event;
                                    }
                                }
                                if text_open { text_open = false; yield ModelEvent::TextEnd; }
                            }
                            Some("function_call") => {
                                saw_completed_item = true;
                                for event in reply_events(reply_split.close_for_tool(), &mut text_open, &mut text, &mut thinking_open) {
                                    yield event;
                                }
                                if thinking_open { thinking_open = false; yield ModelEvent::ThinkingEnd; }
                                let index = required_u64(&value, "output_index")? as usize;
                                saw_tool = true;
                                let owned = json_text(item.get("arguments"));
                                let arguments = owned
                                    .as_deref()
                                    .map_or(ResponseToolArguments::None, ResponseToolArguments::Snapshot);
                                for event in update_response_tool(index, item, arguments, true, &mut tools)? { yield event; }
                            }
                            _ => {}
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        let index = required_u64(&value, "output_index")? as usize;
                        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                            saw_tool = true;
                            for event in update_response_tool(index, &Value::Null, ResponseToolArguments::Delta(delta), false, &mut tools)? { yield event; }
                        }
                    }
                    "response.function_call_arguments.done" => {
                        let index = required_u64(&value, "output_index")? as usize;
                        let owned = json_text(value.get("arguments"));
                        match owned.as_deref() {
                            Some("") => {
                                for event in update_response_tool(
                                    index,
                                    &Value::Null,
                                    ResponseToolArguments::None,
                                    false,
                                    &mut tools,
                                )? { yield event; }
                            }
                            arguments => {
                                let arguments = arguments.map_or(
                                    ResponseToolArguments::None,
                                    ResponseToolArguments::Snapshot,
                                );
                                for event in update_response_tool(index, &Value::Null, arguments, true, &mut tools)? { yield event; }
                            }
                        }
                    }
                    "response.completed" | "response.incomplete" => {
                        let response = value.get("response").unwrap_or(&Value::Null);
                        if let Some(usage) = response.get("usage") { yield ModelEvent::Usage(responses_usage(usage)); }
                        for event in unseen_reasoning_events(response, &mut reasoning_items, &mut thinking_open) { yield event; }
                        if thinking_open { thinking_open = false; yield ModelEvent::ThinkingEnd; }
                        if text_open { text_open = false; yield ModelEvent::TextEnd; }
                        for (index, tool) in tools.iter_mut().filter(|(_, tool)| tool.started && !tool.ended) {
                            if let Some(delta) = take_unsent_arguments(tool, true) {
                                yield ModelEvent::ToolCallArgumentsDelta { index: *index, delta };
                            }
                            tool.ended = true;
                            yield ModelEvent::ToolCallEnd { index: *index };
                        }
                        if kind == "response.completed" && tools.values().any(|tool| !tool.started) {
                            Err(Error::Provider("OpenAI Responses completed with incomplete tool metadata".into()))?;
                        }
                        terminal = true;
                        if !reasoning_items.is_empty() {
                            yield ModelEvent::ProviderReplayState(
                                crate::model::ProviderReplayState {
                                    provider_kind: "openai_responses".into(),
                                    value: json!({"items": std::mem::take(&mut reasoning_items)}),
                                },
                            );
                        }
                        yield ModelEvent::Done(if saw_tool {
                            FinishReason::ToolUse
                        } else if kind == "response.incomplete" {
                            FinishReason::Length
                        } else {
                            FinishReason::Stop
                        });
                    }
                    _ => {}
                }
            }
            if !terminal && saw_completed_item {
                if thinking_open { yield ModelEvent::ThinkingEnd; }
                if text_open { yield ModelEvent::TextEnd; }
                if tools.values().any(|tool| !tool.ended) {
                    Err(Error::Provider("OpenAI Responses stream ended with an incomplete tool call".into()))?;
                }
                terminal = true;
                if !reasoning_items.is_empty() {
                    yield ModelEvent::ProviderReplayState(crate::model::ProviderReplayState {
                        provider_kind: "openai_responses".into(),
                        value: json!({"items": std::mem::take(&mut reasoning_items)}),
                    });
                }
                yield ModelEvent::Done(if saw_tool { FinishReason::ToolUse } else { FinishReason::Stop });
            }
            if !terminal {
                Err(Error::Provider("OpenAI Responses stream ended without response.completed or response.incomplete".into()))?;
            }
        })
    }
}

#[derive(Default)]
struct ReplySplit {
    raw: String,
    held: String,
    phase: ReplyPhase,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum ReplyPhase {
    #[default]
    Probe,
    Thinking,
    Answer,
}

enum ReplyPiece {
    Thinking(String),
    Answer(String),
}

impl ReplySplit {
    fn push(&mut self, delta: &str) -> Vec<ReplyPiece> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.raw.push_str(delta);
        match self.phase {
            ReplyPhase::Answer => vec![ReplyPiece::Answer(delta.to_owned())],
            ReplyPhase::Probe => {
                self.held.push_str(delta);
                self.decide_probe(false)
            }
            ReplyPhase::Thinking => {
                self.held.push_str(delta);
                self.drain_thinking(false)
            }
        }
    }

    fn finish(&mut self) -> Vec<ReplyPiece> {
        match self.phase {
            ReplyPhase::Probe => self.decide_probe(true),
            ReplyPhase::Thinking => self.drain_thinking(true),
            ReplyPhase::Answer => Vec::new(),
        }
    }

    fn close_for_tool(&mut self) -> Vec<ReplyPiece> {
        let pieces = self.finish();
        self.phase = ReplyPhase::Answer;
        self.held.clear();
        pieces
    }

    fn decide_probe(&mut self, force: bool) -> Vec<ReplyPiece> {
        let first_line_done = self.held.contains('\n');
        if !force
            && !first_line_done
            && could_be_thinking_heading(&self.held)
            && self.held.len() < 80
        {
            return Vec::new();
        }
        if is_thinking_heading(&self.held) {
            self.phase = ReplyPhase::Thinking;
            return self.drain_thinking(force);
        }
        self.phase = ReplyPhase::Answer;
        let held = std::mem::take(&mut self.held);
        if held.is_empty() {
            Vec::new()
        } else {
            vec![ReplyPiece::Answer(held)]
        }
    }

    fn drain_thinking(&mut self, force: bool) -> Vec<ReplyPiece> {
        if let Some(at) = answer_boundary(&self.held, force) {
            let thinking = self.held[..at].to_owned();
            let answer = self.held[at..].to_owned();
            self.held.clear();
            self.phase = ReplyPhase::Answer;
            let mut pieces = Vec::new();
            if !thinking.is_empty() {
                pieces.push(ReplyPiece::Thinking(thinking));
            }
            if !answer.is_empty() {
                pieces.push(ReplyPiece::Answer(answer));
            }
            return pieces;
        }
        let hold = thinking_holdback(&self.held, force);
        let emit_len = self.held.len().saturating_sub(hold);
        if emit_len == 0 {
            return Vec::new();
        }
        let emit = self.held[..emit_len].to_owned();
        self.held.replace_range(..emit_len, "");
        vec![ReplyPiece::Thinking(emit)]
    }
}

fn accept_output_text(
    split: &mut ReplySplit,
    text_open: &mut bool,
    text: &mut String,
    thinking_open: &mut bool,
    final_text: &str,
) -> Vec<ModelEvent> {
    let extra = if final_text.starts_with(&split.raw) {
        &final_text[split.raw.len()..]
    } else if split.raw.is_empty() {
        final_text
    } else {
        return reconcile_response_text(text_open, text, final_text);
    };
    reply_events(split.push(extra), text_open, text, thinking_open)
}

fn reply_events(
    pieces: Vec<ReplyPiece>,
    text_open: &mut bool,
    text: &mut String,
    thinking_open: &mut bool,
) -> Vec<ModelEvent> {
    let mut events = Vec::new();
    for piece in pieces {
        match piece {
            ReplyPiece::Thinking(chunk) => {
                if *text_open {
                    *text_open = false;
                    events.push(ModelEvent::TextEnd);
                }
                if !*thinking_open {
                    *thinking_open = true;
                    events.push(ModelEvent::ThinkingStart);
                }
                events.push(ModelEvent::ThinkingDelta(chunk));
            }
            ReplyPiece::Answer(chunk) => {
                if *thinking_open {
                    *thinking_open = false;
                    events.push(ModelEvent::ThinkingEnd);
                }
                if !*text_open {
                    *text_open = true;
                    events.push(ModelEvent::TextStart);
                }
                text.push_str(&chunk);
                events.push(ModelEvent::TextDelta(chunk));
            }
        }
    }
    events
}

fn could_be_thinking_heading(text: &str) -> bool {
    let head = heading_line(text).to_ascii_lowercase();
    let head = head.trim_end_matches(':').trim();
    "thinking process".starts_with(head) || head.starts_with("thinking process")
}

fn is_thinking_heading(text: &str) -> bool {
    heading_line(text)
        .trim_end_matches(':')
        .trim()
        .eq_ignore_ascii_case("thinking process")
}

fn heading_line(text: &str) -> &str {
    text.trim_start()
        .trim_start_matches('#')
        .trim_start()
        .lines()
        .next()
        .unwrap_or("")
        .trim()
}

fn is_thinking_continuation(line: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return true;
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
        || lower.starts_with("wait")
        || lower.starts_with("let's")
        || lower.starts_with("lets ")
        || lower.starts_with("first")
}

fn answer_boundary(text: &str, allow_incomplete_line: bool) -> Option<usize> {
    let mut search = 0;
    while let Some(rel) = text[search..].find("\n\n") {
        let at = search + rel;
        let rest = &text[at + 2..];
        let line = rest.lines().next().unwrap_or("");
        if line.is_empty() {
            search = at + 2;
            continue;
        }
        if !allow_incomplete_line && !rest.contains('\n') {
            return None;
        }
        if is_thinking_continuation(line) {
            search = at + 2;
            continue;
        }
        return Some(at + 2);
    }
    None
}

fn thinking_holdback(text: &str, force: bool) -> usize {
    if force {
        return 0;
    }
    if let Some(at) = text.rfind("\n\n") {
        return text.len() - at;
    }
    if text.ends_with('\n') {
        1
    } else {
        0
    }
}

fn note_reasoning_delta(
    event: &Value,
    indexes: &mut std::collections::BTreeSet<usize>,
    unindexed: &mut bool,
) {
    if let Some(index) = event.get("output_index").and_then(Value::as_u64) {
        indexes.insert(index as usize);
    } else {
        *unindexed = true;
    }
}

fn take_reasoning_item_streamed(
    event: &Value,
    indexes: &mut std::collections::BTreeSet<usize>,
    unindexed: &mut bool,
) -> bool {
    if let Some(index) = event.get("output_index").and_then(Value::as_u64) {
        let index = index as usize;
        if indexes.remove(&index) {
            return true;
        }
        if *unindexed && indexes.is_empty() {
            *unindexed = false;
            return true;
        }
        return false;
    }
    let streamed = *unindexed || !indexes.is_empty();
    *unindexed = false;
    indexes.clear();
    streamed
}

fn requests_reasoning_summary(model: &crate::model::ModelSpec) -> bool {
    model.reasoning.enabled
        || model.reasoning.effort.is_some()
        || model_id_requests_reasoning_summary(&model.model_id)
}

fn model_id_requests_reasoning_summary(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    [
        "gemini", "gpt-5", "gpt-6", "gpt5", "gpt6", "grok", "claude", "opus", "sonnet", "o1", "o3",
        "o4",
    ]
    .into_iter()
    .any(|marker| model_id_has_marker(&model_id, marker))
}

fn model_id_has_marker(model_id: &str, marker: &str) -> bool {
    model_id.match_indices(marker).any(|(index, _)| {
        let before = index == 0 || !model_id.as_bytes()[index - 1].is_ascii_alphanumeric();
        let end = index + marker.len();
        let after = end == model_id.len() || !model_id.as_bytes()[end].is_ascii_alphanumeric();
        before && after
    })
}

fn reasoning_visible_text(item: &Value) -> String {
    let mut pieces = Vec::new();
    match item.get("summary") {
        Some(Value::String(text)) if !text.is_empty() => pieces.push(text.clone()),
        Some(Value::Array(summary)) => {
            for part in summary {
                if let Some(text) = summary_piece_text(part) {
                    pieces.push(text);
                }
            }
        }
        _ => {}
    }
    if let Some(content) = item.get("content").and_then(Value::as_array) {
        for part in content {
            if let Some(text) = content_piece_text(part) {
                pieces.push(text);
            }
        }
    }
    if let Some(text) = item
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        pieces.push(text.to_owned());
    }
    pieces.join("\n")
}

fn unseen_reasoning_events(
    response: &Value,
    seen: &mut Vec<Value>,
    thinking_open: &mut bool,
) -> Vec<ModelEvent> {
    let Some(items) = response.get("output").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut events = Vec::new();
    for item in items {
        let text = reasoning_visible_text(item);
        if text.is_empty() {
            continue;
        }
        let id = item.get("id").and_then(Value::as_str);
        let already = seen.iter().any(|previous| {
            let same_id = id.is_some() && previous.get("id").and_then(Value::as_str) == id;
            let same_text = reasoning_visible_text(previous) == text;
            same_id || same_text
        });
        if already {
            continue;
        }
        if !*thinking_open {
            *thinking_open = true;
            events.push(ModelEvent::ThinkingStart);
            events.push(ModelEvent::ThinkingDelta(text));
        }
        seen.push(item.clone());
    }
    events
}

fn summary_piece_text(part: &Value) -> Option<String> {
    let object = part.as_object()?;
    non_empty_text(object.get("text")).or_else(|| {
        object.get("summary_text").and_then(|summary| {
            non_empty_text(Some(summary)).or_else(|| non_empty_text(summary.get("text")))
        })
    })
}

fn content_piece_text(part: &Value) -> Option<String> {
    if let Some(text) = part.as_str().filter(|text| !text.is_empty()) {
        return Some(text.to_owned());
    }
    let object = part.as_object()?;
    match object.get("type").and_then(Value::as_str) {
        Some("thinking" | "reasoning" | "reasoning_text" | "summary_text") => non_empty_text(
            object
                .get("thinking")
                .or_else(|| object.get("text"))
                .or_else(|| object.get("summary_text")),
        ),
        Some("text") => non_empty_text(object.get("text")),
        _ => non_empty_text(object.get("thinking")),
    }
}

fn non_empty_text(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

fn response_item_text(item: &Value) -> Option<String> {
    let text = item
        .get("content")?
        .as_array()?
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    Some(text)
}

fn reconcile_response_text(
    open: &mut bool,
    streamed: &mut String,
    final_text: &str,
) -> Vec<ModelEvent> {
    let mut events = Vec::new();
    if final_text.starts_with(streamed.as_str()) && final_text.len() > streamed.len() {
        if !*open {
            *open = true;
            events.push(ModelEvent::TextStart);
        }
        let suffix = &final_text[streamed.len()..];
        streamed.push_str(suffix);
        events.push(ModelEvent::TextDelta(suffix.into()));
    }
    events
}

fn remember_announced_arguments(tool: &mut ResponseToolState, item: &Value) {
    let Some(text) = json_text(item.get("arguments")) else {
        return;
    };
    if tool.announced_arguments.is_none() && arguments_have_fields(&text) {
        tool.announced_arguments = Some(text);
    }
}

fn arguments_have_fields(text: &str) -> bool {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| value.as_object().map(|object| !object.is_empty()))
        .unwrap_or(false)
}

fn take_unsent_arguments(tool: &mut ResponseToolState, finish: bool) -> Option<String> {
    if finish && !arguments_have_fields(&tool.arguments) {
        if let Some(announced) = tool
            .announced_arguments
            .clone()
            .filter(|text| arguments_have_fields(text))
        {
            tool.arguments = announced;
            tool.emitted_arguments = 0;
        }
    }
    if tool.emitted_arguments >= tool.arguments.len() {
        return None;
    }
    let delta = tool.arguments[tool.emitted_arguments..].to_string();
    tool.emitted_arguments = tool.arguments.len();
    Some(delta)
}

fn update_response_tool(
    index: usize,
    item: &Value,
    arguments: ResponseToolArguments<'_>,
    done: bool,
    tools: &mut std::collections::BTreeMap<usize, ResponseToolState>,
) -> Result<Vec<ModelEvent>> {
    let tool = tools.entry(index).or_default();
    remember_announced_arguments(tool, item);
    if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
        tool.call_id.get_or_insert_with(|| call_id.into());
    }
    if let Some(name) = item.get("name").and_then(Value::as_str) {
        tool.name.get_or_insert_with(|| name.into());
    }
    match arguments {
        ResponseToolArguments::None => {}
        ResponseToolArguments::Delta(delta) => tool.arguments.push_str(delta),
        ResponseToolArguments::Snapshot(snapshot) if snapshot == tool.arguments => {}
        ResponseToolArguments::Snapshot(snapshot) if snapshot.starts_with(&tool.arguments) => {
            tool.arguments.push_str(&snapshot[tool.arguments.len()..]);
        }
        ResponseToolArguments::Snapshot(snapshot)
            if json_values_match(&tool.arguments, snapshot) => {}
        ResponseToolArguments::Snapshot(_) => {
            return Err(Error::Provider(
                "OpenAI Responses final tool arguments do not match streamed arguments".into(),
            ));
        }
    }

    let mut events = Vec::new();
    if !tool.started {
        if let (Some(call_id), Some(name)) = (&tool.call_id, &tool.name) {
            tool.started = true;
            events.push(ModelEvent::ToolCallStart {
                index,
                call_id: call_id.clone(),
                name: name.clone(),
            });
        }
    }
    if tool.started {
        if let Some(delta) = take_unsent_arguments(tool, done) {
            events.push(ModelEvent::ToolCallArgumentsDelta { index, delta });
        }
    }
    if done && !tool.ended {
        if !tool.started {
            return Err(Error::Provider(
                "OpenAI Responses function call is missing call_id or name".into(),
            ));
        }
        tool.ended = true;
        events.push(ModelEvent::ToolCallEnd { index });
    }
    Ok(events)
}

fn apply_model(
    body: &mut Value,
    model: &crate::model::ModelSpec,
    route_max_output_tokens: Option<u64>,
) -> Result<()> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| Error::Provider("OpenAI Responses request body is not an object".into()))?;
    if let Some(max) = model.max_output_tokens.or(route_max_output_tokens) {
        object.insert("max_output_tokens".into(), json!(max));
    }
    if requests_reasoning_summary(model) {
        let mut reasoning = Map::new();
        reasoning.insert("summary".into(), json!("auto"));
        if let Some(effort) = &model.reasoning.effort {
            reasoning.insert("effort".into(), json!(effort));
        }
        object.insert("reasoning".into(), Value::Object(reasoning));
    }
    if model.latency == ModelLatency::Fast {
        object.insert("service_tier".into(), json!("fast"));
    }
    Ok(())
}

fn responses_input(messages: &[ProjectedMessage]) -> Result<Vec<Value>> {
    let mut input = Vec::new();
    for message in messages {
        match &message.content {
            ProjectedContent::Parts(parts) => {
                push_responses_parts(&mut input, &message.role, parts)?
            }
            ProjectedContent::ToolResult(result) => {
                let output = if result.provider_parts.is_empty() {
                    Value::String(result.content.clone())
                } else {
                    Value::Array(responses_content(&result.provider_parts, "input_text")?)
                };
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": output,
                }));
            }
            ProjectedContent::Assistant {
                text,
                replay_state,
                calls,
                ..
            } => {
                if let Some(state) = replay_state
                    .as_ref()
                    .filter(|state| state.provider_kind == "openai_responses")
                {
                    let items = state
                        .value
                        .get("items")
                        .and_then(Value::as_array)
                        .ok_or_else(|| {
                            Error::Protocol("OpenAI Responses replay state is missing items".into())
                        })?;
                    input.extend(
                        items
                            .iter()
                            .map(response_reasoning_input)
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                push_responses_text(&mut input, &message.role, text);
                for call in calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": serde_json::to_string(&call.arguments)?,
                    }));
                }
            }
        }
    }
    Ok(input)
}

fn response_reasoning_input(item: &Value) -> Result<Value> {
    let source = item
        .as_object()
        .filter(|object| object.get("type").and_then(Value::as_str) == Some("reasoning"))
        .ok_or_else(|| {
            Error::Protocol("OpenAI Responses replay state contains a non-reasoning item".into())
        })?;
    let mut projected = Map::new();
    projected.insert("type".into(), json!("reasoning"));
    for field in ["id", "summary", "content", "encrypted_content"] {
        if let Some(value) = source.get(field) {
            projected.insert(field.into(), value.clone());
        }
    }
    Ok(Value::Object(projected))
}

fn push_responses_parts(input: &mut Vec<Value>, role: &Role, parts: &[ContentPart]) -> Result<()> {
    let text_type = if *role == Role::Assistant {
        "output_text"
    } else {
        "input_text"
    };
    let content = responses_content(parts, text_type)?;
    if !content.is_empty() {
        input.push(json!({
            "type":"message",
            "role":role_name(role),
            "content":content,
        }));
    }
    Ok(())
}

fn responses_content(parts: &[ContentPart], text_type: &str) -> Result<Vec<Value>> {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } if text.is_empty() => None,
            ContentPart::Text { text } => Some(Ok(json!({"type":text_type, "text":text}))),
            ContentPart::Image { mime_type, data } => Some(Ok(json!({
                "type":"input_image",
                "detail":"auto",
                "image_url":format!("data:{mime_type};base64,{}", STANDARD.encode(data)),
            }))),
        })
        .collect()
}

fn push_responses_text(input: &mut Vec<Value>, role: &Role, text: &str) {
    if text.is_empty() {
        return;
    }
    let content_type = if *role == Role::Assistant {
        "output_text"
    } else {
        "input_text"
    };
    input.push(json!({
        "type": "message",
        "role": role_name(role),
        "content": [{"type": content_type, "text": text}],
    }));
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn json_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) => Some(text.clone()),
        value @ (Value::Object(_) | Value::Array(_)) => serde_json::to_string(value).ok(),
        _ => None,
    }
}

fn json_values_match(left: &str, right: &str) -> bool {
    match (
        serde_json::from_str::<Value>(left),
        serde_json::from_str::<Value>(right),
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn required_u64(value: &Value, name: &str) -> Result<u64> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Provider(format!("OpenAI Responses event is missing {name}")))
}

pub(crate) fn responses_usage(value: &Value) -> Usage {
    let input_tokens = value.get("input_tokens").and_then(Value::as_u64);
    Usage {
        input_tokens,
        context_input_tokens: input_tokens,
        output_tokens: value.get("output_tokens").and_then(Value::as_u64),
        total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        cache_read_tokens: super::first_reported_u64(
            value,
            &[
                "/input_tokens_details/cached_tokens",
                "/prompt_tokens_details/cached_tokens",
            ],
        ),
        cache_write_tokens: super::first_reported_u64(
            value,
            &[
                "/input_tokens_details/cache_write_tokens",
                "/input_tokens_details/cache_creation_tokens",
                "/input_tokens_details/cache_creation_input_tokens",
                "/cache_creation_input_tokens",
                "/cache_write_tokens",
            ],
        ),
        reasoning_tokens: value
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelSpec, ProviderReplayState};
    use crate::provider::ModelEvent;

    #[test]
    fn thinking_process_prefix_becomes_thought_and_the_answer_stays_text() {
        let mut split = ReplySplit::default();
        let mut events = split.push("Thinking Process:\n1. Look at the file.\n\n");
        events.extend(split.push("可以开始改了。"));
        events.extend(split.finish());
        let thinking = events
            .iter()
            .filter_map(|piece| match piece {
                ReplyPiece::Thinking(text) => Some(text.as_str()),
                ReplyPiece::Answer(_) => None,
            })
            .collect::<String>();
        let answer = events
            .iter()
            .filter_map(|piece| match piece {
                ReplyPiece::Answer(text) => Some(text.as_str()),
                ReplyPiece::Thinking(_) => None,
            })
            .collect::<String>();
        assert!(thinking.contains("Thinking Process"));
        assert!(thinking.contains("Look at the file"));
        assert_eq!(answer, "可以开始改了。");
    }

    #[test]
    fn ordinary_reply_stays_answer_text() {
        let mut split = ReplySplit::default();
        let events = split.push("你好！有什么需要我帮忙的？");
        assert!(
            matches!(events.as_slice(), [ReplyPiece::Answer(text)] if text == "你好！有什么需要我帮忙的？")
        );
    }

    #[test]
    fn completed_response_output_keeps_reasoning_summary() {
        let response = json!({
            "output": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": "先看用户只是在打招呼"
            }, {
                "type": "message",
                "content": [{"type": "output_text", "text": "你好！"}]
            }]
        });
        let mut seen = Vec::new();
        let mut thinking_open = false;
        let events = unseen_reasoning_events(&response, &mut seen, &mut thinking_open);
        assert!(thinking_open);
        assert!(events.iter().any(|event| matches!(
            event,
            ModelEvent::ThinkingDelta(text) if text == "先看用户只是在打招呼"
        )));
        let again = unseen_reasoning_events(&response, &mut seen, &mut thinking_open);
        assert!(again.is_empty());
    }

    #[test]
    fn object_function_arguments_reach_the_tool_call() {
        let item = json!({
            "call_id": "call-1",
            "name": "Read",
            "arguments": {"path": "C:/repo/file.txt"}
        });
        let owned = json_text(item.get("arguments")).expect("object arguments");
        let mut tools = std::collections::BTreeMap::new();
        let events = update_response_tool(
            0,
            &item,
            ResponseToolArguments::Snapshot(&owned),
            true,
            &mut tools,
        )
        .unwrap();
        let delta = events.iter().find_map(|event| match event {
            ModelEvent::ToolCallArgumentsDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        });
        assert_eq!(delta, Some(owned.as_str()));
        assert!(events
            .iter()
            .any(|event| matches!(event, ModelEvent::ToolCallEnd { index: 0 })));
    }

    #[test]
    fn announced_function_arguments_survive_an_empty_argument_done() {
        let item = json!({
            "call_id": "call-1",
            "name": "Read",
            "arguments": {"path": "/tmp/file.py", "offset": 1, "limit": 300}
        });
        let mut tools = std::collections::BTreeMap::new();
        update_response_tool(0, &item, ResponseToolArguments::None, false, &mut tools).unwrap();
        let events = update_response_tool(
            0,
            &Value::Null,
            ResponseToolArguments::Snapshot("{}"),
            true,
            &mut tools,
        )
        .unwrap();
        let delta = events.iter().find_map(|event| match event {
            ModelEvent::ToolCallArgumentsDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        });
        assert_eq!(
            delta,
            Some(r#"{"limit":300,"offset":1,"path":"/tmp/file.py"}"#)
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, ModelEvent::ToolCallEnd { index: 0 })));
    }

    #[test]
    fn responses_usage_records_gateway_cache_write() {
        let usage = responses_usage(&json!({
            "input_tokens": 1000,
            "output_tokens": 10,
            "total_tokens": 1010,
            "input_tokens_details": {
                "cached_tokens": 128,
                "cache_write_tokens": 512
            }
        }));

        assert_eq!(usage.cache_read_tokens, Some(128));
        assert_eq!(usage.cache_write_tokens, Some(512));
    }

    #[test]
    fn responses_usage_prefers_a_later_positive_cache_write() {
        let usage = responses_usage(&json!({
            "input_tokens": 1000,
            "input_tokens_details": { "cache_write_tokens": 0 },
            "cache_creation_input_tokens": 512
        }));

        assert_eq!(usage.cache_write_tokens, Some(512));
    }

    #[test]
    fn reasoning_stays_requested_when_enabled_without_a_known_name() {
        let mut model = ModelSpec::new("plain-chat");
        model.reasoning.enabled = true;
        model.reasoning.effort = None;
        let mut body = json!({});
        apply_model(&mut body, &model, None).unwrap();
        assert_eq!(body["reasoning"], json!({ "summary": "auto" }));
    }

    #[test]
    fn reasoning_name_match_uses_marker_boundaries() {
        assert!(model_id_requests_reasoning_summary("gpt-6.1-sol"));
        assert!(model_id_requests_reasoning_summary("o1-preview"));
        assert!(!model_id_requests_reasoning_summary("foo1"));
        assert!(!model_id_requests_reasoning_summary("plain-chat"));
    }

    #[test]
    fn responses_usage_leaves_cache_write_empty_when_unreported() {
        let usage = responses_usage(&json!({
            "input_tokens": 51295,
            "output_tokens": 26,
            "input_tokens_details": { "cached_tokens": 0 }
        }));

        assert_eq!(usage.cache_read_tokens, Some(0));
        assert!(usage.cache_write_tokens.is_none());
    }

    #[test]
    fn apply_model_requests_summary_for_gemini_without_effort() {
        let mut model = ModelSpec::new("gemini-4-argon");
        model.reasoning.enabled = false;
        model.reasoning.effort = None;
        let mut body = json!({});
        apply_model(&mut body, &model, None).unwrap();
        assert_eq!(body["reasoning"], json!({ "summary": "auto" }));
        assert!(body["reasoning"].get("effort").is_none());
    }

    #[test]
    fn apply_model_omits_reasoning_for_plain_chat() {
        let mut model = ModelSpec::new("plain-chat");
        model.reasoning.enabled = false;
        model.reasoning.effort = None;
        let mut body = json!({});
        apply_model(&mut body, &model, None).unwrap();
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn message_thinking_field_is_separate_from_the_reply() {
        let item = json!({
            "type": "message",
            "content": [
                {"type": "thinking", "thinking": "先确认用户在打招呼"},
                {"type": "output_text", "text": "你好！"}
            ]
        });
        assert_eq!(reasoning_visible_text(&item), "先确认用户在打招呼");
        assert_eq!(response_item_text(&item).as_deref(), Some("你好！"));
    }

    #[test]
    fn reasoning_visible_text_reads_summary_text() {
        let item = json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "why"}]
        });
        assert_eq!(reasoning_visible_text(&item), "why");
    }

    #[test]
    fn reasoning_replay_projects_response_items_to_valid_input_items() {
        let messages = [ProjectedMessage {
            message_id: "assistant-1".into(),
            role: Role::Assistant,
            content: ProjectedContent::Assistant {
                text: String::new(),
                thinking: String::new(),
                replay_state: Some(ProviderReplayState {
                    provider_kind: "openai_responses".into(),
                    value: json!({
                        "items": [{
                            "type": "reasoning",
                            "id": "item-1",
                            "status": "completed",
                            "summary": [{"type": "summary_text", "text": "why"}],
                            "content": [],
                            "encrypted_content": "opaque",
                            "output_only": true
                        }]
                    }),
                }),
                calls: Vec::new(),
            },
        }];

        assert_eq!(
            responses_input(&messages).unwrap(),
            vec![json!({
                "type": "reasoning",
                "id": "item-1",
                "summary": [{"type": "summary_text", "text": "why"}],
                "content": [],
                "encrypted_content": "opaque"
            })]
        );
    }
}
