//! Layer 3: Token budget injection.
//!
//! When a trace is projected to exceed the configured token budget, a
//! `<budget>` directive is prepended to the system prompt, nudging the
//! model to be more concise without hard-blocking the request.

/// Token budget configuration.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Maximum tokens allowed for the entire trace.
    pub max_tokens: u32,
    /// Fraction of budget consumed before injection kicks in (default: 0.75).
    pub injection_threshold: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        BudgetConfig {
            max_tokens: 8_000,
            injection_threshold: 0.75,
        }
    }
}

impl BudgetConfig {
    /// Returns true if the token budget header should be injected.
    pub fn should_inject(&self, tokens_used: u32) -> bool {
        let ratio = tokens_used as f64 / self.max_tokens as f64;
        ratio >= self.injection_threshold
    }
}

/// Prepend a budget directive to the system prompt.
///
/// Example output:
/// ```text
/// <budget remaining="3200" total="8000">
/// Be concise. Avoid repeating information already established.
/// </budget>
///
/// [original system prompt...]
/// ```
pub fn inject_budget_header(system_prompt: &str, max_tokens: u32, tokens_used: u32) -> String {
    let remaining = max_tokens.saturating_sub(tokens_used);
    let header = format!(
        "<budget remaining=\"{remaining}\" total=\"{max_tokens}\">\n\
         Be concise. Avoid repeating context already established in this conversation.\n\
         </budget>\n\n"
    );
    format!("{header}{system_prompt}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_at_threshold() {
        let cfg = BudgetConfig { max_tokens: 8000, injection_threshold: 0.75 };
        assert!(!cfg.should_inject(5999));
        assert!(cfg.should_inject(6000));
        assert!(cfg.should_inject(7999));
    }

    #[test]
    fn test_inject_header_content() {
        let prompt = inject_budget_header("You are a helpful assistant.", 8000, 6500);
        assert!(prompt.contains("<budget remaining=\"1500\""));
        assert!(prompt.contains("Be concise"));
        assert!(prompt.ends_with("You are a helpful assistant."));
    }
}
