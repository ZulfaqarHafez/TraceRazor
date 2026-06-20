//! MinHash + LSH near-duplicate index.
//!
//! Provides an approximate-nearest-neighbour index over step text so that
//! redundancy detection (SRR) can find candidate near-duplicate pairs in
//! roughly `O(n·K)` time instead of an all-pairs `O(n²)` scan.
//!
//! Pipeline: text → word bigrams (shingles) → 128-value MinHash signature →
//! banded LSH buckets. Two steps that collide in any band become a candidate
//! pair, which is then verified with the signature-based Jaccard estimate.
//!
//! Everything here is deterministic and dependency-free so the core crate stays
//! offline and reproducible.

use std::collections::HashMap;

const NUM_HASHES: usize = 128;
const BAND_SIZE: usize = 4;
const NUM_BANDS: usize = NUM_HASHES / BAND_SIZE;

/// Compute word bigrams from text (lowercased, split on whitespace).
///
/// Texts shorter than two words fall back to single-word shingles so a
/// one-word step still produces a usable signature.
pub fn word_bigrams(text: &str) -> Vec<u64> {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = lowered.split_whitespace().collect();
    if words.len() < 2 {
        return words.iter().map(|w| fnv1a(w.as_bytes())).collect();
    }
    words
        .windows(2)
        .map(|w| fnv1a(format!("{} {}", w[0], w[1]).as_bytes()))
        .collect()
}

/// FNV-1a 64-bit hash.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// Universal hash function for MinHash: `h_i(x) = a_i * x + b_i` (mod 2^64).
fn minhash_val(seed: u64, value: u64) -> u64 {
    let a = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x517c_c1b7_2722_0a95);
    let b = seed
        .wrapping_mul(0x6c62_272e_07bb_0142)
        .wrapping_add(0xd6e8_feb8_6659_fd93);
    a.wrapping_mul(value).wrapping_add(b)
}

/// Compute a MinHash signature (128 values) for a set of shingles.
pub fn minhash_signature(shingles: &[u64]) -> [u64; NUM_HASHES] {
    let mut sig = [u64::MAX; NUM_HASHES];
    for &shingle in shingles {
        for (i, slot) in sig.iter_mut().enumerate() {
            let h = minhash_val(i as u64 + 1, shingle);
            if h < *slot {
                *slot = h;
            }
        }
    }
    sig
}

/// Estimate Jaccard similarity from two MinHash signatures.
pub fn jaccard_estimate(sig_a: &[u64; NUM_HASHES], sig_b: &[u64; NUM_HASHES]) -> f64 {
    let matches = sig_a.iter().zip(sig_b.iter()).filter(|(a, b)| a == b).count();
    matches as f64 / NUM_HASHES as f64
}

/// Serialise a `u64` slice to little-endian bytes (safe alternative to a raw
/// pointer cast) so it can be fed to [`fnv1a`].
fn sig_to_bytes(sig: &[u64]) -> Vec<u8> {
    sig.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// LSH index: stores step signatures and returns candidate near-duplicate
/// pairs with an estimated Jaccard at or above the configured threshold.
pub struct LshIndex {
    /// One map per band: `band_hash -> step indices that hashed to it`.
    buckets: Vec<HashMap<u64, Vec<usize>>>,
    signatures: Vec<[u64; NUM_HASHES]>,
    threshold: f64,
}

impl LshIndex {
    /// Create an empty index that keeps candidate pairs estimated at Jaccard
    /// `>= threshold`.
    pub fn new(threshold: f64) -> Self {
        Self {
            buckets: (0..NUM_BANDS).map(|_| HashMap::new()).collect(),
            signatures: Vec::new(),
            threshold,
        }
    }

    /// Insert a step's text under `step_idx`. Steps should be inserted in
    /// ascending index order so `candidate_pairs` returns `(i, j)` with `i < j`.
    pub fn insert(&mut self, step_idx: usize, text: &str) {
        let shingles = word_bigrams(text);
        let sig = minhash_signature(&shingles);
        for (band, map) in self.buckets.iter_mut().enumerate() {
            let band_sig = &sig[band * BAND_SIZE..(band + 1) * BAND_SIZE];
            let band_hash = fnv1a(&sig_to_bytes(band_sig));
            map.entry(band_hash).or_default().push(step_idx);
        }
        self.signatures.push(sig);
    }

    /// Return candidate pairs `(a, b)` with `a < b` and estimated Jaccard
    /// `>= threshold`.
    pub fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        let mut seen = std::collections::HashSet::new();
        let mut pairs = Vec::new();
        for band_buckets in &self.buckets {
            for bucket in band_buckets.values() {
                if bucket.len() < 2 {
                    continue;
                }
                for i in 0..bucket.len() {
                    for j in i + 1..bucket.len() {
                        let a = bucket[i].min(bucket[j]);
                        let b = bucket[i].max(bucket[j]);
                        if a != b && seen.insert((a, b)) {
                            let est = jaccard_estimate(&self.signatures[a], &self.signatures[b]);
                            if est >= self.threshold {
                                pairs.push((a, b));
                            }
                        }
                    }
                }
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_has_jaccard_one() {
        let s = minhash_signature(&word_bigrams("the quick brown fox jumps"));
        assert!((jaccard_estimate(&s, &s) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn disjoint_text_has_low_jaccard() {
        let a = minhash_signature(&word_bigrams("alpha beta gamma delta epsilon"));
        let b = minhash_signature(&word_bigrams("ninety nine red balloons floating"));
        assert!(jaccard_estimate(&a, &b) < 0.2);
    }

    #[test]
    fn near_duplicates_become_candidate_pairs() {
        let mut idx = LshIndex::new(0.5);
        idx.insert(0, "search flights from new york to seattle today");
        idx.insert(1, "completely unrelated text about cooking pasta");
        idx.insert(2, "search flights from new york to seattle today");
        let pairs = idx.candidate_pairs();
        assert!(
            pairs.contains(&(0, 2)),
            "exact duplicate steps 0 and 2 should be a candidate pair, got {pairs:?}"
        );
        assert!(
            !pairs.contains(&(0, 1)),
            "unrelated steps should not be a candidate pair, got {pairs:?}"
        );
    }

    #[test]
    fn case_insensitive_shingling() {
        let a = minhash_signature(&word_bigrams("Hello World Foo"));
        let b = minhash_signature(&word_bigrams("hello world foo"));
        assert!((jaccard_estimate(&a, &b) - 1.0).abs() < 1e-9);
    }
}
