//! Verifies captured Cursor Connect framing and protobuf compatibility.
#[path = "support/fake_cursor.rs"]
mod fake_cursor;
#[path = "support/fake_provider.rs"]
mod fake_provider;
#[path = "support/fixtures.rs"]
mod fixtures;

use std::{io::Write, sync::Arc};

use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
};
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine};
use cursor_server::{
    api::cursor,
    cursor::prompting::{PromptAssets, PromptCompiler},
    cursor::protocol::{
        connect,
        proto::{agent::v1 as pb, aiserver::v1 as ai},
    },
    cursor::transport::TransportRegistry,
    network::NetworkClients,
};
use flate2::{write::GzEncoder, Compression};
use prost::Message;
use tower::ServiceExt;

#[test]
fn connect_envelope_is_flag_plus_big_endian_length_plus_protobuf() {
    let message = pb::BidiRequestId {
        request_id: "abc".into(),
    };
    let frame = connect::encode_message(&message).unwrap();
    assert_eq!(&frame[..5], &[0, 0, 0, 0, 5]);
    let decoded: pb::BidiRequestId = fake_cursor::decode_single(&frame).unwrap();
    assert_eq!(decoded.request_id, "abc");
}

#[test]
fn end_stream_matches_captured_connect_shape() {
    assert_eq!(
        connect::encode_end_stream().as_ref(),
        &[2, 0, 0, 0, 2, b'{', b'}']
    );
}

#[test]
fn error_end_stream_is_flagged_json_not_protobuf() {
    let frame = connect::encode_error_end_stream(&connect::ConnectStreamError {
        code: connect::ConnectCode::Unavailable,
        message: "overloaded".into(),
        details: vec![connect::ConnectErrorDetail {
            type_name: "aiserver.v1.ErrorDetails".into(),
            value: "AQ".into(),
        }],
    })
    .unwrap();
    let (flags, payload) = connect::decode_frames(&frame).unwrap().pop().unwrap();
    assert_eq!(flags, connect::END_STREAM_FLAG);
    let json: serde_json::Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(json["error"]["code"], "unavailable");
    assert_eq!(json["error"]["message"], "overloaded");
    assert_eq!(
        json["error"]["details"][0]["type"],
        "aiserver.v1.ErrorDetails"
    );
}

#[test]
fn cursor_error_details_subset_decodes_captured_wire_value() {
    let captured = "CAISVQoUQXV0aGVudGljYXRpb24gZXJyb3ISMklmIHlvdSBhcmUgbG9nZ2VkIGluLCB0cnkgbG9nZ2luZyBvdXQgYW5kIGJhY2sgaW4uIABSBwoFbG9naW4YAQ";
    let bytes = STANDARD_NO_PAD.decode(captured).unwrap();
    let details = ai::ErrorDetails::decode(bytes.as_slice()).unwrap();
    assert_eq!(details.error, 2, "ERROR_NOT_LOGGED_IN");
    assert_eq!(details.is_expected, Some(true));
    let custom = details.details.unwrap();
    assert_eq!(custom.title, "Authentication error");
    assert_eq!(custom.is_retryable, Some(false));
}

#[test]
fn captured_kv_ack_hex_decodes_as_agent_client_message() {
    let bytes = hex::decode("1a0408011a00").unwrap();
    let message = pb::AgentClientMessage::decode(bytes.as_slice()).unwrap();
    let Some(pb::agent_client_message::Message::KvClientMessage(kv)) = message.message else {
        panic!("expected KV client message")
    };
    assert_eq!(kv.id, 1);
    assert!(matches!(
        kv.message,
        Some(pb::kv_client_message::Message::SetBlobResult(_))
    ));
}

#[tokio::test]
async fn bidi_append_gzip_body_is_decompressed_before_protobuf_decode() {
    let (_directory, store) = fixtures::temp_store().await;
    let assets = PromptAssets::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("prompt/cursor")
            .as_path(),
    )
    .unwrap();
    let clients = NetworkClients::new(store.clone());
    let registry = TransportRegistry::new(
        store,
        Arc::new(fake_provider::FakeProvider::default()),
        PromptCompiler::new(assets),
    );
    let wire = ai::BidiAppendRequest {
        request_id: Some(ai::BidiRequestId {
            request_id: "gzip-request".into(),
        }),
        ..Default::default()
    }
    .encode_to_vec();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&wire).unwrap();
    let compressed = encoder.finish().unwrap();

    let response = cursor::router(registry, clients)
        .unwrap()
        .oneshot(
            Request::post("/aiserver.v1.BidiService/BidiAppend")
                .header(header::CONTENT_TYPE, "application/proto")
                .header(header::CONTENT_ENCODING, "gzip")
                .body(Body::from(compressed))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("BidiAppend contains no AgentClientMessage"));
    assert!(!text.contains("protobuf decode error"));
}

fn run_with_state(state: pb::ConversationStateStructure) -> pb::AgentClientMessage {
    pb::AgentClientMessage {
        message: Some(pb::agent_client_message::Message::RunRequest(
            pb::AgentRunRequest {
                conversation_state: Some(state),
                ..Default::default()
            },
        )),
    }
}

fn append_request(payload: Vec<u8>, binary: bool) -> ai::BidiAppendRequest {
    ai::BidiAppendRequest {
        data: if binary {
            String::new()
        } else {
            hex::encode(&payload)
        },
        request_id: Some(ai::BidiRequestId {
            request_id: "req".into(),
        }),
        append_seqno: 0,
        data_binary: if binary { payload } else { Vec::new() },
    }
}

fn summary_of(message: pb::AgentClientMessage) -> Option<Vec<u8>> {
    let Some(pb::agent_client_message::Message::RunRequest(request)) = message.message else {
        panic!("expected run request");
    };
    request.conversation_state.unwrap().summary
}

#[test]
fn blank_and_existing_conversation_state_roundtrip_through_hex_and_binary() {
    let blank = run_with_state(pb::ConversationStateStructure::default());
    let blank_bytes = blank.encode_to_vec();
    let decoded = cursor::bidi::decode(&append_request(blank_bytes, false)).unwrap();
    assert_eq!(summary_of(decoded.message), None);

    let existing = run_with_state(pb::ConversationStateStructure {
        turns: vec![b"turn-blob".to_vec()],
        summary: Some(b"summary-blob".to_vec()),
        ..Default::default()
    });
    let existing_bytes = existing.encode_to_vec();
    for binary in [false, true] {
        let decoded =
            cursor::bidi::decode(&append_request(existing_bytes.clone(), binary)).unwrap();
        assert_eq!(
            summary_of(decoded.message).as_deref(),
            Some(b"summary-blob".as_slice())
        );
    }
}

#[test]
fn fixed32_summary_keeps_the_wire_mismatch_and_omits_payload_bytes() {
    let state = fixed32_field(6, 0x1122_3344);
    let run = length_delimited(1, &state);
    let message = length_delimited(1, &run);
    let error = cursor::bidi::decode(&append_request(message, false))
        .unwrap_err()
        .to_string();
    assert!(error.contains("ThirtyTwoBit"));
    assert!(error.contains("#6/i32"));
    assert!(!error.contains("11223344"));
    assert!(!error.contains("44332211"));
    assert!(!error.contains("summary-blob"));
}

fn length_delimited(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = proto_key(field, 2);
    out.extend(proto_varint(payload.len() as u64));
    out.extend(payload);
    out
}

fn fixed32_field(field: u32, value: u32) -> Vec<u8> {
    let mut out = proto_key(field, 5);
    out.extend(value.to_le_bytes());
    out
}

fn proto_key(field: u32, wire_type: u32) -> Vec<u8> {
    proto_varint(u64::from((field << 3) | wire_type))
}

fn proto_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return out;
        }
    }
}
