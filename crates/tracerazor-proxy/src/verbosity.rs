//! Proxy Layer 4 — Verbosity Directive Injection
//!
//! Triggered when the caller reports a rolling average CCR (Caveman Compression
//! Ratio) of the last three steps at or above 0.35. Injects a conciseness
//! directive into the outgoing system prompt to nudge the model toward shorter,
//! denser output.
//!
//! Two severity levels:
//! - Standard (rolling_ccr 0.35–0.50): strip preamble and hedging
//! - Ultra (rolling_ccr > 0.50): maximally concise; single-sentence reasoning
//!
//! The injection is a non-blocking soft nudge — requests are never blocked by
//! this layer.

/// CCR threshold that activates the standard verbosity directive.
pub const STANDARD_THRESHOLD: f64 = 0.35;

/// CCR threshold that escalates to the ultra-concise directive.
pub const ULTRA_THRESHOLD: f64 = 0.50;

/// The directive injected at standard severity.
pub const STANDARD_DIRECTIVE: &str =
    "\n\n[VERBOSITY DIRECTIVE] Your recent responses have shown high compressibility. \
     Avoid preamble sentences (\"Let me...\", \"I'd be happy to...\"), hedging phrases \
     (might, could, perhaps), and filler adverbs (basically, actually, essentially). \
     State your reasoning directly and concisely.";

/// The directive injected at ultra severity.
pub const ULTRA_DIRECTIVE: &str =
    "\n\n[VERBOSITY DIRECTIVE — ULTRA] Your recent responses are highly redundant and \
     compressible. Be maximally concise: one sentence per reasoning step where possible. \
     Skip all preamble, hedging, and throat-clearing. Output only the minimum information \
     needed to advance the task.";

/// Inject the appropriate verbosity directive into `system_prompt` based on `rolling_ccr`.
///
/// Returns the (possibly modified) system prompt. If `rolling_ccr` is `None` or below
/// the standard threshold, returns the prompt unchanged.
pub fn inject_verbosity_directive(system_prompt: &str, rolling_ccr: Option<f64>) -> String {
    match rolling_ccr {
        Some(ccr) if ccr > ULTRA_THRESHOLD => {
            format!("{}{}", system_prompt, ULTRA_DIRECTIVE)
        }
        Some(ccr) if ccr >= STANDARD_THRESHOLD => {
            format!("{}{}", system_prompt, STANDARD_DIRECTIVE)
        }
        _ => system_prompt.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_injection_below_threshold() {
        let prompt = "You are a helpful assistant.";
        let result = inject_verbosity_directive(prompt, Some(0.20));
        assert_eq!(result, prompt, "below threshold: no injection");
    }

    #[test]
    fn test_no_injection_when_none() {
        let prompt = "You are a helpful assistant.";
        let result = inject_verbosity_directive(prompt, None);
        assert_eq!(result, prompt, "None rolling_ccr: no injection");
    }

    #[test]
    fn test_standard_injection_at_threshold() {
        let prompt = "You are a helpful assistant.";
        let result = inject_verbosity_directive(prompt, Some(0.35));
        assert!(
            result.contains("[VERBOSITY DIRECTIVE]"),
            "CCR at 0.35 should inject standard directive"
        );
        assert!(
            !result.contains("ULTRA"),
            "standard severity should not include ULTRA"
        );
    }

    #[test]
    fn test_standard_injection_mid_range() {
        let prompt = "You are a helpful assistant.";
        let result = inject_verbosity_directive(prompt, Some(0.45));
        assert!(result.contains("[VERBOSITY DIRECTIVE]"));
        assert!(!result.contains("ULTRA"));
    }

    #[test]
    fn test_ultra_injection_above_threshold() {
        let prompt = "You are a helpful assistant.";
        let result = inject_verbosity_directive(prompt, Some(0.55));
        assert!(
            result.contains("ULTRA"),
            "CCR > 0.50 should inject ultra directive"
        );
    }

    #[test]
    fn test_ultra_injection_exact_boundary() {
        let prompt = "System prompt.";
        let result = inject_verbosity_directive(prompt, Some(0.501));
        assert!(result.contains("ULTRA"));
    }

    #[test]
    fn test_original_prompt_preserved() {
        let prompt = "Custom system instructions here.";
        let result = inject_verbosity_directive(prompt, Some(0.40));
        assert!(
            result.starts_with(prompt),
            "directive is appended, not prepended"
        );
    }
}