//! Sends one provider HTTP request and keeps the connection-layer cause.

use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

use super::CallRecorder;

#[derive(Debug)]
pub(crate) enum Attempt {
    Response(reqwest::Response),
    Cancelled,
}

pub(crate) async fn send_once<F>(
    label: &str,
    build: F,
    cancellation: &CancellationToken,
    recorder: Option<&CallRecorder>,
) -> Result<Attempt>
where
    F: FnOnce() -> reqwest::RequestBuilder,
{
    let response = tokio::select! {
        _ = cancellation.cancelled() => return Ok(Attempt::Cancelled),
        response = build().send() => response,
    };
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return Err(Error::Provider(format!(
                "{label}: {}",
                transport_detail(&error)
            )));
        }
    };
    if let Some(recorder) = recorder {
        recorder
            .response_headers(response.status().as_u16())
            .await?;
    }
    if response.status().is_success() {
        return Ok(Attempt::Response(response));
    }
    let status = response.status();
    let bytes = tokio::select! {
        _ = cancellation.cancelled() => return Ok(Attempt::Cancelled),
        bytes = response.bytes() => bytes,
    }?;
    Err(Error::Provider(format!(
        "{label} {status}: {}",
        String::from_utf8_lossy(&bytes)
    )))
}

fn transport_detail(error: &reqwest::Error) -> String {
    let mut detail = error.to_string();
    let mut current: &(dyn std::error::Error + 'static) = error;
    while let Some(source) = current.source() {
        let text = source.to_string();
        if !detail.contains(&text) {
            detail.push_str(": ");
            detail.push_str(&text);
        }
        current = source;
    }
    detail
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn local_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    async fn server(response: &'static [u8]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request).await;
            socket.write_all(response).await.unwrap();
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn non_success_status_is_one_failed_attempt() {
        let url =
            server(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\n\r\ndown").await;
        let client = local_client();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::Provider(message) if message.contains("503") && message.contains("down"))
        );
    }

    #[tokio::test]
    async fn response_body_transport_failure_is_one_failed_attempt() {
        let url =
            server(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\n\r\nshort").await;
        let client = local_client();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }

    #[tokio::test]
    async fn request_transport_failure_reports_the_connection_cause() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{address}");
        let client = local_client();
        let error = send_once("test", || client.get(&url), &CancellationToken::new(), None)
            .await
            .unwrap_err();
        let Error::Provider(message) = error else {
            panic!("expected the connection cause");
        };
        assert!(message.starts_with("test: "), "{message}");
        assert!(
            message.contains("error sending request") || message.contains("connection"),
            "{message}"
        );
    }
}
