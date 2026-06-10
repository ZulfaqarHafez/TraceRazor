/// Semantic similarity engine for TraceRazor.
///
/// Phase 1: Term-frequency bag-of-words cosine similarity (fully offline, no API key).
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

/// Boxed, thread-safe similarity closure (the async/embedding API surface).
pub type BoxedSimilarityFn = Box<dyn Fn(&str, &str) -> f64 + Send + Sync>;

/// Trait for any similarity backend.
pub trait Similarity: Send + Sync {
    /// Returns cosine similarity between two texts in [0.0, 1.0].
    fn similarity(&self, a: &str, b: &str) -> f64;
}

/// Backend identity for the offline bag-of-words similarity.
pub const BOW_BACKEND_ID: &str = "bow";

/// Phase 1 default: bag-of-words cosine similarity (no API key required).
///
/// The closure memoises per-text TF vectors: the metric layer presents the
/// same step texts to the similarity function thousands of times inside
/// pairwise windows, so each distinct text is tokenised exactly once.
/// Results are identical to the uncached path (same 4-dp rounding).
pub fn default_similarity_fn() -> impl Fn(&str, &str) -> f64 {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    let engine = BowSimilarity::new();
    let cache: RefCell<HashMap<String, Rc<HashMap<String, f64>>>> =
        RefCell::new(HashMap::new());
    move |a: &str, b: &str| {
        let get = |s: &str| -> Rc<HashMap<String, f64>> {
            if let Some(v) = cache.borrow().get(s) {
                return Rc::clone(v);
            }
            let v = Rc::new(engine.tf(s));
            cache.borrow_mut().insert(s.to_owned(), Rc::clone(&v));
            v
        };
        let (ta, tb) = (get(a), get(b));
        BowSimilarity::cosine_tf(&ta, &tb)
    }
}

/// Boxed, thread-safe variant of the memoised BoW closure (for async paths).
fn cached_bow_boxed() -> BoxedSimilarityFn {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    let engine = BowSimilarity::new();
    let cache: Mutex<HashMap<String, Arc<HashMap<String, f64>>>> =
        Mutex::new(HashMap::new());
    Box::new(move |a: &str, b: &str| {
        let get = |s: &str| -> Arc<HashMap<String, f64>> {
            let mut guard = cache.lock().expect("similarity cache poisoned");
            if let Some(v) = guard.get(s) {
                return Arc::clone(v);
            }
            let v = Arc::new(engine.tf(s));
            guard.insert(s.to_owned(), Arc::clone(&v));
            v
        };
        let (ta, tb) = (get(a), get(b));
        BowSimilarity::cosine_tf(&ta, &tb)
    })
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
) -> BoxedSimilarityFn {
    embedding_similarity_fn_with_identity(texts).await.0
}

/// Like [`embedding_similarity_fn`], but also reports which backend the
/// closure actually uses (`"bow"` or `"embeddings:<model>"`). The silent
/// BoW fallbacks become a recorded fact the caller can put in a run
/// manifest instead of an invisible scoring change.
pub async fn embedding_similarity_fn_with_identity(
    texts: Vec<String>,
) -> (BoxedSimilarityFn, String) {
    let Some(cfg) = LlmConfig::from_env() else {
        return (cached_bow_boxed(), BOW_BACKEND_ID.to_string());
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
            let identity = format!("embeddings:{embed_model}");
            let f: BoxedSimilarityFn =
                Box::new(move |a: &str, b: &str| {
                    match (text_index.get(a), text_index.get(b)) {
                        (Some(&i), Some(&j)) => {
                            openai::cosine_similarity(&embeddings[i], &embeddings[j])
                        }
                        _ => bow.similarity(a, b),
                    }
                });
            (f, identity)
        }
        Err(e) => {
            eprintln!(
                "Warning: embeddings backend failed ({e}); falling back to BoW similarity"
            );
            (cached_bow_boxed(), BOW_BACKEND_ID.to_string())
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memoised_closure_matches_uncached_engine_exactly() {
        let cached = default_similarity_fn();
        let engine = BowSimilarity::new();
        let texts = [
            "parse the user request about order refund",
            "parse user request order refund details re-read",
            "execute database query retrieve records",
            "",
            "parse the user request about order refund", // repeat → cache hit
        ];
        for a in &texts {
            for b in &texts {
                assert_eq!(
                    cached(a, b),
                    engine.similarity(a, b),
                    "cache must be invisible for ({a:?}, {b:?})"
                );
            }
        }
    }
}
