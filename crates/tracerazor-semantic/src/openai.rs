/// OpenAI embeddings backend for semantic similarity.
///
/// Uses `text-embedding-3-small` (1536-dim) via the OpenAI API.
/// Replaces the Phase 1 BoW similarity for production use, enabling
/// the PRD-specified 0.85 cosine threshold for SRR.
///
/// Falls back to BowSimilarity when OPENAI_API_KEY is not set.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Request body for the OpenAI embeddings endpoint.
#[derive(Serialize)]
struct EmbedRequest {
    input: Vec<String>,
    model: String,
    encoding_format: &'static str,
}

/// Response from the OpenAI embeddings endpoint.
#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

/// Fetch embeddings for a batch of texts from the OpenAI API.
pub async fn embed_batch(texts: &[String], api_key: &str, model: &str) -> Result<Vec<Vec<f32>>> {
    let client = reqwest::Client::new();

    let request = EmbedRequest {
        input: texts.to_vec(),
        model: model.to_string(),
        encoding_format: "float",
    };

    let response = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .await
        .context("OpenAI API request failed")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI API error {}: {}", status, body);
    }

    let embed_response: EmbedResponse = response
        .json()
        .await
        .context("Failed to parse OpenAI embeddings response")?;

    Ok(embed_response.data.into_iter().map(|d| d.embedding).collect())
}

/// Cosine similarity between two float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)) as f64
}

/// Pre-compute embeddings for all steps in a trace, then return a
/// synchronous similarity closure that uses the cached vectors.
///
/// This is the recommended path for SRR when an API key is available:
/// batch embed all steps in one API call, then run the O(n²) comparison
/// with no additional network calls.
pub async fn build_similarity_cache(
    texts: &[String],
    api_key: &str,
    model: &str,
) -> Result<Vec<Vec<f32>>> {
    embed_batch(texts, api_key, model).await
}
