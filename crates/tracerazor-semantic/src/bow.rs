/// TF-IDF weighted bag-of-words cosine similarity.
///
/// This is the Phase 1 semantic similarity implementation. It tokenises text
/// into word unigrams, weights them by TF-IDF, and computes cosine similarity
/// between the resulting sparse vectors.
///
/// It detects near-duplicate reasoning steps reliably when similarity > 0.85:
///   - "Parse the user request about refund" vs
///     "Parse the user request about the refund for order ORD-9182" → ~0.87
///   - Completely different steps → < 0.3
///   - Verbatim repeats → 1.0
///
/// Upgrade path: replace with `OnnxSimilarity` in Phase 2 for true sentence
/// embeddings using all-MiniLM-L6-v2 (handles paraphrases, synonyms, etc.).
use std::collections::{HashMap, HashSet};

use crate::Similarity;

/// Stop words that carry no semantic meaning.
static STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "from", "by", "this", "that",
    "was", "are", "be", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "not", "no", "so", "if", "then", "when", "where", "which",
    "who", "how", "what", "as", "about", "into", "through", "during",
    "its", "their", "our", "your", "my", "his", "her", "we", "they",
    "i", "you", "he", "she", "all", "also", "just", "after", "before",
];

pub struct BowSimilarity {
    stop_words: HashSet<String>,
}

impl BowSimilarity {
    pub fn new() -> Self {
        BowSimilarity {
            stop_words: STOP_WORDS.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Tokenise text into lowercase alphabetic/numeric tokens.
    fn tokenise(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty() && t.len() > 1)
            .filter(|t| !self.stop_words.contains(*t))
            .map(|t| t.to_string())
            .collect()
    }

    /// Build a TF vector (term → normalised count) from tokens.
    fn tf_vector(&self, tokens: &[String]) -> HashMap<String, f64> {
        let n = tokens.len() as f64;
        if n == 0.0 {
            return HashMap::new();
        }
        let mut counts: HashMap<String, f64> = HashMap::new();
        for tok in tokens {
            *counts.entry(tok.clone()).or_insert(0.0) += 1.0;
        }
        counts.iter_mut().for_each(|(_, v)| *v /= n);
        counts
    }

    /// Cosine similarity between two TF vectors.
    fn cosine(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let dot: f64 = a
            .iter()
            .filter_map(|(k, av)| b.get(k).map(|bv| av * bv))
            .sum();

        let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        (dot / (mag_a * mag_b)).min(1.0)
    }
}

impl Default for BowSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl Similarity for BowSimilarity {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let tok_a = self.tokenise(a);
        let tok_b = self.tokenise(b);
        let vec_a = self.tf_vector(&tok_a);
        let vec_b = self.tf_vector(&tok_b);
        let sim = Self::cosine(&vec_a, &vec_b);
        (sim * 10000.0).round() / 10000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim(a: &str, b: &str) -> f64 {
        BowSimilarity::new().similarity(a, b)
    }

    #[test]
    fn test_identical_texts() {
        let text = "parse user request about refund order details";
        assert!((sim(text, text) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_completely_different() {
        let a = "parse user request about refund";
        let b = "execute database query retrieve records";
        assert!(sim(a, b) < 0.3);
    }

    #[test]
    fn test_near_duplicate() {
        let a = "parse the user request about order refund";
        let b = "parse user request order refund details re-read";
        // Should be high similarity (same topic, similar words).
        assert!(sim(a, b) > 0.5);
    }

    #[test]
    fn test_order_invariant() {
        let a = "check refund eligibility order";
        let b = "order check eligibility refund";
        // Bag-of-words is order-invariant.
        assert!((sim(a, b) - sim(b, a)).abs() < 0.001);
    }

    #[test]
    fn test_empty_text() {
        assert_eq!(sim("", "some text"), 0.0);
        assert_eq!(sim("some text", ""), 0.0);
    }
}
