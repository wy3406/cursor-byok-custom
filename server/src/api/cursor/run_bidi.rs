//! Serves Cursor's HTTP/2 `InferenceService/RunInference` stream.
//!
//! The client sends `RunInferenceClientMessage` frames and reads
//! `RunInferenceServerMessage` frames on the same request. `x-request-id`
//! selects the session. Each data frame uses the Connect envelope: one flag
//! byte, a four-byte big-endian length, then the protobuf message.

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, HeaderValue, Request, Response, StatusCode},
};
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use prost::Message;

use crate::{
    api::cursor::inference,
    cursor::{
        protocol::{
            connect::{self, END_STREAM_FLAG},
            proto::inference::v1 as inf,
        },
        transport::{TransportHandle, TransportParent, TransportRegistry},
    },
    Error, Result,
};

const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;
const COMPRESSION_FLAG: u8 = 0x01;

pub async fn handler(
    State(registry): State<TransportRegistry>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    let (parts, body) = request.into_parts();
    let request_id = request_id(&parts.headers)?;
    let parent = parent_headers(&parts.headers)?;
    tracing::info!(request_id, "accepted inference bidi stream");
    let handle = registry.get_or_create(&request_id).await?;
    let receiver = handle.subscribe();
    let reader_handle = handle.clone();
    let reader = registry.clone();
    let reader_id = request_id;
    tokio::spawn(async move {
        if let Err(error) = read_client_frames(&reader, &reader_id, parent, body).await {
            tracing::warn!(
                request_id = reader_id.as_str(),
                %error,
                "inference bidi read failed"
            );
            let _ = reader_handle.emit_frame(
                connect::encode_error_end_stream(&connect::ConnectStreamError {
                    code: connect::ConnectCode::InvalidArgument,
                    message: error.to_string(),
                    details: Vec::new(),
                })
                .unwrap_or_else(|_| connect::encode_end_stream()),
            );
        }
    });
    let body_stream = async_stream::stream! {
        let _guard = StreamGuard(handle);
        while let Some(chunk) = receiver.recv().await {
            let terminal = chunk.first().is_some_and(|flags| flags & END_STREAM_FLAG != 0);
            yield Ok::<Bytes, std::convert::Infallible>(chunk);
            if terminal {
                return;
            }
        }
    };
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/connect+proto"),
    );
    response
        .headers_mut()
        .insert("connect-protocol-version", HeaderValue::from_static("1"));
    Ok(response)
}

async fn read_client_frames(
    registry: &TransportRegistry,
    request_id: &str,
    parent: Option<TransportParent>,
    body: Body,
) -> Result<()> {
    let mut body = body.into_data_stream();
    let mut buffer = BytesMut::new();
    let mut seqno = 0_i64;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|error| {
            Error::Protocol(format!("cannot read inference bidi body: {error}"))
        })?;
        buffer.extend_from_slice(&chunk);
        while let Some((flags, payload)) = pop_frame(&mut buffer)? {
            if flags & END_STREAM_FLAG != 0 {
                return Ok(());
            }
            if flags & COMPRESSION_FLAG != 0 {
                return Err(Error::Protocol(
                    "compressed inference bidi frames are not accepted".into(),
                ));
            }
            let decoded = inf::RunInferenceClientMessage::decode(payload.as_ref())?;
            let Some(message) = decoded.message else {
                continue;
            };
            inference::accept(registry, request_id, seqno, message, parent.clone()).await?;
            seqno += 1;
        }
    }
    if !buffer.is_empty() {
        return Err(Error::Protocol(
            "inference bidi body ended inside a frame".into(),
        ));
    }
    Ok(())
}

fn pop_frame(buffer: &mut BytesMut) -> Result<Option<(u8, Bytes)>> {
    if buffer.len() < 5 {
        return Ok(None);
    }
    let length = u32::from_be_bytes([buffer[1], buffer[2], buffer[3], buffer[4]]) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(Error::Protocol(
            "inference bidi frame exceeds the size limit".into(),
        ));
    }
    if buffer.len() < 5 + length {
        return Ok(None);
    }
    let flags = buffer[0];
    let _ = buffer.split_to(5);
    Ok(Some((flags, buffer.split_to(length).freeze())))
}

fn request_id(headers: &HeaderMap) -> Result<String> {
    let value = headers
        .get("x-request-id")
        .ok_or_else(|| Error::Protocol("inference bidi x-request-id is required".into()))?;
    let request_id = value
        .to_str()
        .map_err(|error| Error::Protocol(format!("invalid x-request-id header: {error}")))?
        .trim();
    if request_id.is_empty() {
        return Err(Error::Protocol(
            "inference bidi x-request-id is required".into(),
        ));
    }
    Ok(request_id.to_owned())
}

fn parent_headers(headers: &HeaderMap) -> Result<Option<TransportParent>> {
    let request_id = header_text(headers, "x-parent-request-id")?;
    let tool_call_id = header_text(headers, "x-parent-agent-tool-call-id")?;
    match (request_id, tool_call_id) {
        (None, None) => Ok(None),
        (Some(request_id), Some(tool_call_id)) => Ok(Some(TransportParent {
            request_id: request_id.into(),
            tool_call_id: tool_call_id.into(),
        })),
        _ => Err(Error::Protocol(
            "Cursor subagent request must include both parent headers".into(),
        )),
    }
}

fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>> {
    headers
        .get(name)
        .map(|value| value.to_str())
        .transpose()
        .map_err(|error| Error::Protocol(format!("invalid {name} header: {error}")))
}

struct StreamGuard(TransportHandle);

impl Drop for StreamGuard {
    fn drop(&mut self) {
        let handle = self.0.clone();
        tokio::spawn(async move {
            handle.disconnect().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn partial_envelope_stays_buffered() {
        let mut buffer = BytesMut::from(&b"\x00\x00\x00\x00\x02"[..]);
        assert!(pop_frame(&mut buffer).unwrap().is_none());
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn run_request_frame_decodes_after_a_split_chunk() {
        let message = inf::RunInferenceClientMessage {
            message: Some(inf::run_inference_client_message::Message::RunRequest(
                inf::RunInferenceRunRequest {
                    conversation_id: "66666666-6666-4666-8666-666666666666".into(),
                    requested_model: Some(inf::InferenceRequestedModel {
                        model_id: "model-alpha".into(),
                        ..Default::default()
                    }),
                    routing_conversation: Vec::new(),
                },
            )),
        };
        let frame = connect::encode_message(&message).unwrap();
        let mut buffer = BytesMut::from(&frame[..4]);
        assert!(pop_frame(&mut buffer).unwrap().is_none());
        buffer.extend_from_slice(&frame[4..]);
        let (flags, payload) = pop_frame(&mut buffer).unwrap().unwrap();
        assert_eq!(flags, 0);
        assert!(buffer.is_empty());
        let decoded = inf::RunInferenceClientMessage::decode(payload.as_ref()).unwrap();
        let inf::run_inference_client_message::Message::RunRequest(request) =
            decoded.message.unwrap()
        else {
            panic!("expected a run request");
        };
        assert_eq!(
            request.conversation_id,
            "66666666-6666-4666-8666-666666666666"
        );
        assert_eq!(request.requested_model.unwrap().model_id, "model-alpha");
    }
}
