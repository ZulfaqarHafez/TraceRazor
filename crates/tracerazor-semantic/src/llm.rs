//! Pluggable LLM backend for TraceRazor.
//!
//! Supports three provider shapes:
//!   - `openai`            — api.openai.com chat/completions + embeddings
//!   - `anthropic`         — api.anthropic.com messages (no embeddings)
//!   - `openai-compatible` — any endpoint that speaks the OpenAI chat/embeddings
//!     wire format (Ollama, vLLM, Groq, Together, OpenRouter, Azure OpenAI, LM Studio, …)
//!
//! Selection is env-driven:
//!   - `TRACERAZOR_LLM_PROVIDER` = `openai` | `anthropic` | `openai-compatible`
//!   - `TRACERAZOR_LLM_BASE_URL` = full base URL (default: provider-specific)
//!   - `TRACERAZOR_LLM_MODEL`    = model name (default: provider-specific)
//!   - `TRACERAZOR_LLM_API_KEY`  = generic key (overrides provider-specific)
//!
//! Backward compatibility:
//!   - If `TRACERAZOR_LLM_PROVIDER` is unset and `OPENAI_API_KEY` is present,
//!     the provider defaults to `openai`.
//!   - If `ANTHROPIC_API_KEY` is present (and no OpenAI key), it defaults to
//!     `anthropic`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// The wire format the backend speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Hosted OpenAI (`api.openai.com`).
    Openai,
    /// Hosted Anthropic (`api.anthropic.com`).
    Anthropic,
    /// Any OpenAI-compatible endpoint (Ollama, vLLM, Azure, OpenRouter, …).
    OpenaiCompatible,
}

impl Provider {
    fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Provider::Openai),
            "anthropic" | "claude" => Some(Provider::Anthropic),
            "openai-compatible" | "openai_compatible" | "oai-compat" => {
                Some(Provider::OpenaiCompatible)
            }
            _ => None,
        }
    }

    fn default_base(&self) -> &'static str {
        match self {
            Provider::Openai => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com/v1",
            // OpenaiCompatible has no sensible default base — the user must provide one.
            Provider::OpenaiCompatible => "",
        }
    }

    fn default_model(&self) -> &'static str {
        match self {
            Provider::Openai | Provider::OpenaiCompatible => "gpt-4o-mini",
            Provider::Anthropic => "claude-haiku-4-5-20251001",
        }
    }
}

/// Resolved configuration for an LLM backend.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: Provider,
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl LlmConfig {
    /// Build a config by inspecting environment variables.
    ///
    /// Returns `None` when no credentials are available — callers should
    /// fall back to their offline path (e.g. BoW similarity).
    pub fn from_env() -> Option<Self> {
        Self::from_env_map(|k| std::env::var(k).ok())
    }

    /// Same as [`from_env`] but takes an explicit getter so tests can inject
    /// an env map without touching process-global state.
    pub fn from_env_map<F>(get: F) -> Option<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let explicit = get("TRACERAZOR_LLM_PROVIDER")
            .as_deref()
            .and_then(Provider::parse);

        let (provider, api_key) = match explicit {
            Some(Provider::Openai) => (
                Provider::Openai,
                get("TRACERAZOR_LLM_API_KEY")
                    .or_else(|| get("OPENAI_API_KEY"))
                    .unwrap_or_default(),
            ),
            Some(Provider::Anthropic) => (
                Provider::Anthropic,
                get("TRACERAZOR_LLM_API_KEY")
                    .or_else(|| get("ANTHROPIC_API_KEY"))
                    .unwrap_or_default(),
            ),
            Some(Provider::OpenaiCompatible) => (
                Provider::OpenaiCompatible,
                // Local servers (Ollama/vLLM) often need no key — empty string is fine.
                get("TRACERAZOR_LLM_API_KEY")
                    .or_else(|| get("OPENAI_API_KEY"))
                    .unwrap_or_default(),
            ),
            None => {
                // Auto-detect from whichever vendor key is present.
                if let Some(k) = get("OPENAI_API_KEY") {
                    (Provider::Openai, k)
                } else if let Some(k) = get("ANTHROPIC_API_KEY") {
                    (Provider::Anthropic, k)
                } else if get("TRACERAZOR_LLM_BASE_URL").is_some() {
                    // Base URL alone ⇒ assume openai-compatible (typical for local).
                    (
                        Provider::OpenaiCompatible,
                        get("TRACERAZOR_LLM_API_KEY")
                            .or_else(|| get("OPENAI_API_KEY"))
                            .unwrap_or_default(),
                    )
                } else {
                    return None;
                }
            }
        };

        // Openai-compatible requires a base URL; anything else just overrides the default.
        let base_url = get("TRACERAZOR_LLM_BASE_URL")
            .unwrap_or_else(|| provider.default_base().to_string());
        if provider == Provider::OpenaiCompatible && base_url.is_empty() {
            return None;
        }

        let model =
            get("TRACERAZOR_LLM_MODEL").unwrap_or_else(|| provider.default_model().to_string());

        Some(LlmConfig {
            provider,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            api_key,
        })
    }

    /// Manual constructor — primarily used by tests and library consumers who
    /// want to bypass the env-var resolution.
    pub fn new(
        provider: Provider,
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        LlmConfig {
            provider,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Send a single chat/messages request and return the assistant text.
    pub async fn complete(&self, system: &str, user: &str) -> Result<String> {
        match self.provider {
            Provider::Openai | Provider::OpenaiCompatible => {
                complete_openai(&self.base_url, &self.api_key, &self.model, system, user).await
            }
            Provider::Anthropic => {
                complete_anthropic(&self.base_url, &self.api_key, &self.model, system, user).await
            }
        }
    }

    /// Fetch embeddings for a batch of texts.
    ///
    /// Returns an error for Anthropic, which has no embeddings API — callers
    /// should treat this as a signal to fall back to BoW similarity.
    pub async fn embed(&self, texts: &[String], model: &str) -> Result<Vec<Vec<f32>>> {
        match self.provider {
            Provider::Openai | Provider::OpenaiCompatible => {
                embed_openai(&self.base_url, &self.api_key, model, texts).await
            }
            Provider::Anthropic => {
                bail!("Anthropic does not expose an embeddings API; fall back to BoW similarity")
            }
        }
    }
}

// ── OpenAI-shaped chat ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OpenaiChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenaiMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenaiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenaiChatResponse {
    choices: Vec<OpenaiChoice>,
}

#[derive(Deserialize)]
struct OpenaiChoice {
    message: OpenaiChoiceMessage,
}

#[derive(Deserialize)]
struct OpenaiChoiceMessage {
    #[serde(default)]
    content: String,
}

async fn complete_openai(
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Result<String> {
    let client = reqwest::Client::new();
    let body = OpenaiChatRequest {
        model,
        messages: vec![
            OpenaiMessage {
                role: "system",
                content: system,
            },
            OpenaiMessage {
                role: "user",
                content: user,
            },
        ],
        temperature: 0.0,
        max_tokens: 256,
    };

    let url = format!("{}/chat/completions", base_url);
    let mut req = client.post(&url).json(&body);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let response = req
        .send()
        .await
        .with_context(|| format!("chat request to {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let txt = response.text().await.unwrap_or_default();
        bail!("LLM API error {status}: {txt}");
    }

    let chat: OpenaiChatResponse = response
        .json()
        .await
        .context("failed to parse OpenAI-shaped chat response")?;
    Ok(chat
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default())
}

// ── Anthropic messages ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicBlock>,
}

#[derive(Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

async fn complete_anthropic(
    base_url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Result<String> {
    let client = reqwest::Client::new();
    let body = AnthropicRequest {
        model,
        max_tokens: 256,
        system,
        messages: vec![AnthropicMessage {
            role: "user",
            content: user,
        }],
        temperature: 0.0,
    };

    let url = format!("{}/messages", base_url);
    let response = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .with_context(|| format!("messages request to {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let txt = response.text().await.unwrap_or_default();
        bail!("Anthropic API error {status}: {txt}");
    }

    let parsed: AnthropicResponse = response
        .json()
        .await
        .context("failed to parse Anthropic messages response")?;
    let text = parsed
        .content
        .into_iter()
        .filter(|b| b.block_type == "text")
        .map(|b| b.text)
        .collect::<Vec<_>>()
        .join("");
    Ok(text)
}

// ── OpenAI-shaped embeddings ──────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

async fn embed_openai(
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let client = reqwest::Client::new();
    let body = EmbedRequest {
        input: texts,
        model,
        encoding_format: "float",
    };
    let url = format!("{}/embeddings", base_url);
    let mut req = client.post(&url).json(&body);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    let response = req
        .send()
        .await
        .with_context(|| format!("embeddings request to {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let txt = response.text().await.unwrap_or_default();
        bail!("embeddings API error {status}: {txt}");
    }

    let parsed: EmbedResponse = response
        .json()
        .await
        .context("failed to parse embeddings response")?;
    Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── env-map resolution ────────────────────────────────────────────────

    fn map<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn from_env_auto_detects_openai() {
        let cfg = LlmConfig::from_env_map(map(&[("OPENAI_API_KEY", "sk-test")])).unwrap();
        assert_eq!(cfg.provider, Provider::Openai);
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.api_key, "sk-test");
    }

    #[test]
    fn from_env_auto_detects_anthropic() {
        let cfg = LlmConfig::from_env_map(map(&[("ANTHROPIC_API_KEY", "sk-ant-test")])).unwrap();
        assert_eq!(cfg.provider, Provider::Anthropic);
        assert_eq!(cfg.base_url, "https://api.anthropic.com/v1");
        assert!(cfg.model.starts_with("claude-"));
    }

    #[test]
    fn from_env_prefers_openai_when_both_keys_present() {
        let cfg = LlmConfig::from_env_map(map(&[
            ("OPENAI_API_KEY", "sk-openai"),
            ("ANTHROPIC_API_KEY", "sk-ant"),
        ]))
        .unwrap();
        assert_eq!(cfg.provider, Provider::Openai);
    }

    #[test]
    fn from_env_explicit_provider_wins() {
        let cfg = LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "anthropic"),
            ("OPENAI_API_KEY", "sk-openai"),
            ("ANTHROPIC_API_KEY", "sk-ant"),
        ]))
        .unwrap();
        assert_eq!(cfg.provider, Provider::Anthropic);
        assert_eq!(cfg.api_key, "sk-ant");
    }

    #[test]
    fn from_env_openai_compatible_requires_base_url() {
        // Missing base_url → None.
        assert!(LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "openai-compatible")
        ]))
        .is_none());

        // With base_url → OK (empty key allowed for local servers).
        let cfg = LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "openai-compatible"),
            ("TRACERAZOR_LLM_BASE_URL", "http://localhost:11434/v1"),
            ("TRACERAZOR_LLM_MODEL", "llama3.1"),
        ]))
        .unwrap();
        assert_eq!(cfg.provider, Provider::OpenaiCompatible);
        assert_eq!(cfg.base_url, "http://localhost:11434/v1");
        assert_eq!(cfg.model, "llama3.1");
        assert_eq!(cfg.api_key, "");
    }

    #[test]
    fn from_env_bare_base_url_implies_openai_compatible() {
        let cfg = LlmConfig::from_env_map(map(&[(
            "TRACERAZOR_LLM_BASE_URL",
            "http://localhost:8000/v1",
        )]))
        .unwrap();
        assert_eq!(cfg.provider, Provider::OpenaiCompatible);
    }

    #[test]
    fn from_env_returns_none_without_credentials() {
        assert!(LlmConfig::from_env_map(map(&[])).is_none());
    }

    #[test]
    fn from_env_strips_trailing_slash() {
        let cfg = LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "openai-compatible"),
            ("TRACERAZOR_LLM_BASE_URL", "http://localhost:11434/v1/"),
        ]))
        .unwrap();
        assert_eq!(cfg.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn generic_api_key_overrides_vendor_key() {
        let cfg = LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "openai"),
            ("TRACERAZOR_LLM_API_KEY", "sk-generic"),
            ("OPENAI_API_KEY", "sk-vendor"),
        ]))
        .unwrap();
        assert_eq!(cfg.api_key, "sk-generic");
    }

    #[test]
    fn openai_compatible_falls_back_to_openai_key() {
        let cfg = LlmConfig::from_env_map(map(&[
            ("TRACERAZOR_LLM_PROVIDER", "openai-compatible"),
            ("TRACERAZOR_LLM_BASE_URL", "http://localhost:11434/v1"),
            ("OPENAI_API_KEY", "sk-shared"),
        ]))
        .unwrap();
        assert_eq!(cfg.provider, Provider::OpenaiCompatible);
        assert_eq!(cfg.api_key, "sk-shared");
    }

    // ── HTTP wire format (mocked) ────────────────────────────────────────

    #[tokio::test]
    async fn complete_openai_hits_chat_completions() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({"model": "gpt-4o-mini"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "hello from openai"}}]
            })))
            .mount(&server)
            .await;

        let cfg = LlmConfig::new(Provider::Openai, server.uri(), "gpt-4o-mini", "sk-test");
        let out = cfg.complete("be terse", "hi").await.unwrap();
        assert_eq!(out, "hello from openai");
    }

    #[tokio::test]
    async fn complete_anthropic_hits_messages() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_partial_json(
                serde_json::json!({"model": "claude-haiku-4-5-20251001"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "text", "text": "hello from "},
                    {"type": "text", "text": "claude"}
                ]
            })))
            .mount(&server)
            .await;

        let cfg = LlmConfig::new(
            Provider::Anthropic,
            server.uri(),
            "claude-haiku-4-5-20251001",
            "sk-ant-test",
        );
        let out = cfg.complete("be terse", "hi").await.unwrap();
        assert_eq!(out, "hello from claude");
    }

    #[tokio::test]
    async fn complete_openai_compatible_works_without_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "local llama reply"}}]
            })))
            .mount(&server)
            .await;

        let cfg = LlmConfig::new(Provider::OpenaiCompatible, server.uri(), "llama3.1", "");
        let out = cfg.complete("sys", "usr").await.unwrap();
        assert_eq!(out, "local llama reply");
    }

    #[tokio::test]
    async fn embed_hits_embeddings_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]}
                ]
            })))
            .mount(&server)
            .await;

        let cfg = LlmConfig::new(Provider::Openai, server.uri(), "gpt-4o-mini", "sk-test");
        let vecs = cfg
            .embed(
                &["hello".to_string(), "world".to_string()],
                "text-embedding-3-small",
            )
            .await
            .unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(vecs[1], vec![0.4, 0.5, 0.6]);
    }

    #[tokio::test]
    async fn embed_fails_on_anthropic_provider() {
        let cfg = LlmConfig::new(
            Provider::Anthropic,
            "http://unused",
            "claude-haiku-4-5-20251001",
            "sk",
        );
        let err = cfg
            .embed(&["x".to_string()], "text-embedding-3-small")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Anthropic"));
    }

    #[tokio::test]
    async fn complete_surfaces_http_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .mount(&server)
            .await;

        let cfg = LlmConfig::new(Provider::Openai, server.uri(), "gpt-4o-mini", "bad-key");
        let err = cfg.complete("s", "u").await.unwrap_err();
        assert!(err.to_string().contains("401"));
    }
}