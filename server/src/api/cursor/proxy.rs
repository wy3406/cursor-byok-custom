//! Selects local handling or the configured official Cursor upstream.
use std::time::{Duration, Instant};

use axum::{
    body::{to_bytes, Body, Bytes},
    extract::Extension,
    http::{header, Request, Response},
};

use crate::{Error, Result};

const CURSOR_UPSTREAM_HEADER_TIMEOUT: Duration = Duration::from_secs(30);

const CURSOR_UPSTREAM: &str = "https://api2.cursor.sh";
pub const UPSTREAM_URL_HEADER: &str = "x-server-upstream-url";

#[derive(Clone)]
pub struct CursorProxy {
    clients: crate::network::NetworkClients,
    upstream: String,
}

pub struct BufferedResponse {
    pub status: axum::http::StatusCode,
    pub headers: axum::http::HeaderMap,
    pub body: Bytes,
}

impl BufferedResponse {
    pub fn into_response(self) -> Response<Body> {
        let body = self.body.clone();
        self.with_body(body)
    }

    pub fn with_body(mut self, body: Bytes) -> Response<Body> {
        self.headers.insert(
            header::CONTENT_LENGTH,
            body.len()
                .to_string()
                .parse()
                .expect("body length is always a valid header value"),
        );
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = self.status;
        *response.headers_mut() = self.headers;
        response
    }
}

impl CursorProxy {
    pub fn cursor(clients: crate::network::NetworkClients) -> Self {
        Self {
            clients,
            upstream: CURSOR_UPSTREAM.into(),
        }
    }

    async fn client(&self) -> Result<reqwest::Client> {
        self.clients.cursor_client().await
    }
}

pub async fn forward(
    Extension(proxy): Extension<CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    forward_request(&proxy, request, None, CURSOR_UPSTREAM_HEADER_TIMEOUT).await
}

pub(crate) async fn forward_to_service(
    proxy: &CursorProxy,
    request: Request<Body>,
    service_url: &str,
) -> Result<Response<Body>> {
    forward_request(
        proxy,
        request,
        Some(service_url),
        CURSOR_UPSTREAM_HEADER_TIMEOUT,
    )
    .await
}

async fn forward_request(
    proxy: &CursorProxy,
    request: Request<Body>,
    service_url: Option<&str>,
    header_timeout: Duration,
) -> Result<Response<Body>> {
    let started = Instant::now();
    let (parts, body) = request.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or("/", |value| value.as_str())
        .to_owned();
    let url = match service_url {
        Some(service_url) => format!("{}{}", service_url.trim_end_matches('/'), path),
        None => upstream_url(&parts.headers, &proxy.upstream, &path)?,
    };

    let mut headers = parts.headers;
    headers.remove(UPSTREAM_URL_HEADER);
    headers.remove(header::HOST);
    remove_hop_by_hop_headers(&mut headers);

    let client = proxy.client().await?;
    let upstream = send_for_headers(
        client
            .request(parts.method.clone(), url)
            .headers(headers)
            .body(reqwest::Body::wrap_stream(body.into_data_stream())),
        header_timeout,
    )
    .await;

    let upstream = match upstream {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(
                method = %parts.method,
                path,
                elapsed_ms = started.elapsed().as_millis(),
                %error,
                "Cursor upstream request failed"
            );
            return Err(error);
        }
    };

    let status = upstream.status();
    let mut response_headers = upstream.headers().clone();
    remove_hop_by_hop_headers(&mut response_headers);
    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;

    tracing::info!(
        method = %parts.method,
        path,
        %status,
        elapsed_ms = started.elapsed().as_millis(),
        "forwarded Cursor backend request"
    );
    Ok(response)
}

pub async fn forward_buffered(
    proxy: &CursorProxy,
    request: Request<Body>,
) -> Result<BufferedResponse> {
    let (parts, body) = request.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or("/", |value| value.as_str());
    let url = upstream_url(&parts.headers, &proxy.upstream, path)?;
    let mut headers = parts.headers;
    headers.remove(UPSTREAM_URL_HEADER);
    headers.remove(header::HOST);
    remove_hop_by_hop_headers(&mut headers);
    headers.insert(
        "connect-accept-encoding",
        axum::http::HeaderValue::from_static("identity"),
    );
    headers.insert(
        header::ACCEPT_ENCODING,
        axum::http::HeaderValue::from_static("identity"),
    );
    let body = to_bytes(body, usize::MAX)
        .await
        .map_err(|error| crate::Error::Protocol(format!("cannot read request body: {error}")))?;
    let upstream = send_for_headers(
        proxy
            .client()
            .await?
            .request(parts.method, url)
            .headers(headers)
            .body(body),
        CURSOR_UPSTREAM_HEADER_TIMEOUT,
    )
    .await?;
    let status = upstream.status();
    let mut headers = upstream.headers().clone();
    remove_hop_by_hop_headers(&mut headers);
    let body = read_upstream_body(upstream, CURSOR_UPSTREAM_HEADER_TIMEOUT).await?;
    Ok(BufferedResponse {
        status,
        headers,
        body,
    })
}

async fn send_for_headers(
    request: reqwest::RequestBuilder,
    header_timeout: Duration,
) -> Result<reqwest::Response> {
    match tokio::time::timeout(header_timeout, request.send()).await {
        Ok(result) => result.map_err(Into::into),
        Err(_) => Err(upstream_timeout("did not respond", header_timeout)),
    }
}

async fn read_upstream_body(response: reqwest::Response, timeout: Duration) -> Result<Bytes> {
    match tokio::time::timeout(timeout, response.bytes()).await {
        Ok(result) => Ok(result?),
        Err(_) => Err(upstream_timeout("did not finish the response", timeout)),
    }
}

fn upstream_timeout(action: &str, timeout: Duration) -> Error {
    Error::Provider(format!(
        "Cursor upstream {action} within {} seconds",
        timeout.as_secs().max(1)
    ))
}

fn upstream_url(headers: &axum::http::HeaderMap, fallback: &str, path: &str) -> Result<String> {
    let Some(value) = headers.get(UPSTREAM_URL_HEADER) else {
        return Ok(format!("{fallback}{path}"));
    };
    let value = value
        .to_str()
        .map_err(|error| crate::Error::Protocol(format!("invalid upstream URL header: {error}")))?;
    let url = reqwest::Url::parse(value)
        .map_err(|error| crate::Error::Protocol(format!("invalid upstream URL: {error}")))?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "https" || !crate::local_app::proxy_host_allowed(host) {
        return Err(crate::Error::Protocol(
            "upstream URL must target a Cursor HTTPS host".into(),
        ));
    }
    Ok(url.into())
}

fn remove_hop_by_hop_headers(headers: &mut axum::http::HeaderMap) {
    let connection_headers = headers
        .get(header::CONNECTION)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        headers.remove(name);
    }
    headers.remove("keep-alive");
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Body, http::Request};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{forward_request, CursorProxy};
    use crate::{network::NetworkClients, store::Store, Error};

    async fn proxy() -> (tempfile::TempDir, CursorProxy) {
        let directory = tempfile::tempdir().unwrap();
        let url = format!(
            "sqlite://{}",
            directory.path().join("proxy-timeout.db").display()
        );
        let store = Store::connect(&url).await.unwrap();
        let proxy = CursorProxy::cursor(NetworkClients::new(store));
        (directory, proxy)
    }

    #[tokio::test]
    async fn silent_cursor_upstream_stops_at_the_header_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
            drop(socket);
        });
        let (_directory, proxy) = proxy().await;
        let request = Request::builder()
            .method("POST")
            .uri("/aiserver.v1.DashboardService/GetTeams")
            .body(Body::empty())
            .unwrap();
        let outcome = tokio::time::timeout(
            Duration::from_secs(3),
            forward_request(
                &proxy,
                request,
                Some(&format!("http://{address}")),
                Duration::from_millis(200),
            ),
        )
        .await;
        let error = outcome
            .expect("a silent Cursor upstream ends at the header timeout")
            .expect_err("a silent upstream is an error");
        assert!(
            matches!(error, Error::Provider(ref message) if message.contains("did not respond")),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_prompt_cursor_upstream_response_is_returned() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0_u8; 4096];
            let mut seen = 0_usize;
            loop {
                let read = socket.read(&mut buffer[seen..]).await.unwrap();
                if read == 0 {
                    return;
                }
                seen += read;
                if buffer[..seen]
                    .windows(4)
                    .any(|window| window == b"\r\n\r\n")
                {
                    break;
                }
            }
            socket
                .write_all(
                    b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });
        let (_directory, proxy) = proxy().await;
        let request = Request::builder()
            .uri("/aiserver.v1.DashboardService/GetTeams")
            .body(Body::empty())
            .unwrap();
        let response = forward_request(
            &proxy,
            request,
            Some(&format!("http://{address}")),
            Duration::from_secs(3),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
    }
}
