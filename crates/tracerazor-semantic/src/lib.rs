/// Semantic similarity engine for TraceRazor.
///
/// Phase 1: TF-IDF bag-of-words cosine similarity (fully offline, no API key).
/// Phase 2: Dense sentence embeddings via a pluggable LLM backend — OpenAI,
///          Anthropic (chat only), or any OpenAI-compatible endpoint (Ollama,
///          vLLM, Azure OpenAI, OpenRouter, Groq, Together, LM Studio, …).
///
/// Backend selection is controlled by `tracerazor_semantic::llm::LlmConfig`,
/// which reads `TRACERAZOR_LLM_PROVIDER` / `TRACERAZOR_LLM_BASE_URL` /
/// `TRACERAZOR_LLM_MODEL` / `TRACERAZOR_LLM_API_KEY` from the environment,
/// with graceful fallback to `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
pub mod bow;
pub mod llm;
pub mod openai;

pub use bow::BowSimilarity;
pub use llm::{LlmConfig, Provider};

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

/// Phase 2: Build a similarity closure backed by pre-computed embeddings from
/// whichever LLM backend is configured via the environment.
///
/// Fetches all embeddings in a single batched API call, then returns a closure
/// that computes cosine similarity from the cached vectors — no additional
/// network calls during the O(n²) step comparison.
///
/// Falls back to BoW similarity if:
///   - no credentials are present,
///   - the configured provider has no embeddings API (Anthropic), or
///   - the embeddings request fails for any reason.
pub async fn embedding_similarity_fn(
    texts: Vec<String>,
) -> Box<dyn Fn(&str, &str) -> f64 + Send + Sync> {
    let Some(cfg) = LlmConfig::from_env() else {
        let engine = BowSimilarity::new();
        return Box::new(move |a, b| engine.similarity(a, b));
    };

    let embed_model = std::env::var("TRACERAZOR_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());

    match cfg.embed(&texts, &embed_model).await {
        Ok(embeddings) => {
            let text_index: std::collections::HashMap<String, usize> = texts
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect();

            let bow = BowSimilarity::new();
            Box::new(move |a: &str, b: &str| {
                match (text_index.get(a), text_index.get(b)) {
                    (Some(&i), Some(&j)) => {
                        openai::cosine_similarity(&embeddings[i], &embeddings[j])
                    }
                    _ => bow.similarity(a, b),
                }
            })
        }
        Err(e) => {
            eprintln!(
                "Warning: embeddings backend failed ({e}); falling back to BoW similarity"
            );
            let engine = BowSimilarity::new();
            Box::new(move |a, b| engine.similarity(a, b))
        }
    }
}

/// Backward-compatible alias for the old OpenAI-only helper.
/// Prefer [`embedding_similarity_fn`] in new code.
#[deprecated(note = "Renamed to `embedding_similarity_fn` now that other backends are supported")]
pub async fn openai_similarity_fn(
    texts: Vec<String>,
) -> Box<dyn Fn(&str, &str) -> f64 + Send + Sync> {
    embedding_similarity_fn(texts).await
}