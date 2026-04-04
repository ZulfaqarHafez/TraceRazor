/// Semantic similarity engine for TraceRazor.
///
/// Phase 1: TF-IDF bag-of-words cosine similarity (fully offline, no API needed).
/// Phase 2: OpenAI `text-embedding-3-small` for true sentence-level similarity,
///          enabling the PRD-specified 0.85 cosine threshold.
///
/// The engine auto-selects the backend based on OPENAI_API_KEY availability.
pub mod bow;
pub mod openai;

pub use bow::BowSimilarity;

/// Trait for any similarity backend.
pub trait Similarity: Send + Sync {
    /// Returns cosine similarity between two texts in [0.0, 1.0].
    fn similarity(&self, a: &str, b: &str) -> f64;
}

/// Phase 1 default: bag-of-words cosine similarity (no API key required).
pub fn default_similarity_fn() -> impl Fn(&str, &str) -> f64 {
    let engine = BowSimilarity::new();
    move |a: &str, b: &str| engine.similarity(a, b)
}

/// Phase 2: Build a similarity closure backed by pre-computed OpenAI embeddings.
///
/// Fetches all embeddings in a single batched API call, then returns a closure
/// that computes cosine similarity from the cached vectors — no additional
/// network calls during the O(n²) step comparison.
///
/// Falls back to BoW similarity if the API call fails.
pub async fn openai_similarity_fn(
    texts: Vec<String>,
) -> Box<dyn Fn(&str, &str) -> f64 + Send + Sync> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let model = std::env::var("TRACERAZOR_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());

    if api_key.is_empty() {
        let engine = BowSimilarity::new();
        return Box::new(move |a, b| engine.similarity(a, b));
    }

    match openai::build_similarity_cache(&texts, &api_key, &model).await {
        Ok(embeddings) => {
            // Build a text→index lookup for the closure.
            let text_index: std::collections::HashMap<String, usize> = texts
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect();

            let bow = BowSimilarity::new();
            Box::new(move |a: &str, b: &str| {
                let idx_a = text_index.get(a);
                let idx_b = text_index.get(b);
                match (idx_a, idx_b) {
                    (Some(&i), Some(&j)) => {
                        openai::cosine_similarity(&embeddings[i], &embeddings[j])
                    }
                    _ => bow.similarity(a, b),
                }
            })
        }
        Err(e) => {
            eprintln!("Warning: OpenAI embeddings failed ({}), falling back to BoW", e);
            let engine = BowSimilarity::new();
            Box::new(move |a, b| engine.similarity(a, b))
        }
    }
}

/// LLM chat completion client for RDA and DBO metrics.
pub mod llm {
    use anyhow::{Context, Result};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct ChatRequest {
        model: String,
        messages: Vec<Message>,
        temperature: f32,
        max_tokens: u32,
    }

    #[derive(Serialize, Deserialize)]
    struct Message {
        role: String,
        content: String,
    }

    #[derive(Deserialize)]
    struct ChatResponse {
        choices: Vec<Choice>,
    }

    #[derive(Deserialize)]
    struct Choice {
        message: Message,
    }

    /// Send a single chat completion request and return the text response.
    pub async fn complete(system: &str, user: &str) -> Result<String> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY not set")?;
        let model = std::env::var("TRACERAZOR_LLM_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let client = reqwest::Client::new();
        let request = ChatRequest {
            model,
            messages: vec![
                Message { role: "system".into(), content: system.to_string() },
                Message { role: "user".into(), content: user.to_string() },
            ],
            temperature: 0.0,
            max_tokens: 256,
        };

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&request)
            .send()
            .await
            .context("OpenAI chat request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API error {}: {}", status, body);
        }

        let chat: ChatResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI chat response")?;

        Ok(chat
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }
}
