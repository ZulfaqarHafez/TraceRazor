/// Semantic similarity engine for TraceRazor.
///
/// Phase 1 implements TF-IDF bag-of-words cosine similarity — accurate enough for
/// detecting near-duplicate reasoning steps (97% similar parse requests, etc.) and
/// runs fully offline with zero dependencies.
///
/// Phase 2 upgrade path: swap `BowSimilarity` for `OnnxSimilarity` which uses
/// `ort` + all-MiniLM-L6-v2 for true sentence-level semantic similarity.
pub mod bow;

pub use bow::BowSimilarity;

/// Trait for any similarity backend.
pub trait Similarity: Send + Sync {
    /// Returns cosine similarity between two texts in the range [0.0, 1.0].
    fn similarity(&self, a: &str, b: &str) -> f64;
}

/// Get a closure wrapping the default Phase 1 similarity backend.
/// This is the function injected into `tracerazor_core::analyse`.
pub fn default_similarity_fn() -> impl Fn(&str, &str) -> f64 {
    let engine = BowSimilarity::new();
    move |a: &str, b: &str| engine.similarity(a, b)
}
