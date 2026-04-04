/// WebSocket handler for real-time trace events.
///
/// Connect at `ws://localhost:8080/ws` to receive JSON events as traces
/// are analysed. Useful for live dashboard updates without polling.
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};

use crate::state::{AppState, WsEvent};

pub async fn handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.events.subscribe();

    // Task: forward broadcast events → WebSocket client.
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let json = match serde_json::to_string(&event) {
                        Ok(j) => j,
                        Err(_) => continue,
                    };
                    if sender.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(_) => continue, // lagged — skip old events
            }
        }
    });

    // Drain incoming messages (keep-alive pings) until client disconnects.
    while let Some(Ok(_)) = receiver.next().await {}

    send_task.abort();
}
