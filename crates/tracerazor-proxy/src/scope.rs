//! Layer 2: Scope whitelist — blocks disallowed tool calls.

/// Scope whitelist configuration.
#[derive(Debug, Clone, Default)]
pub struct ScopeConfig {
    /// Set of allowed tool names. If empty, all tools are allowed.
    pub allowed_tools: Vec<String>,
    /// Set of explicitly blocked tool names (checked even if allowed_tools is empty).
    pub blocked_tools: Vec<String>,
}

impl ScopeConfig {
    /// Create a scope that only allows the listed tools.
    pub fn whitelist(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        ScopeConfig {
            allowed_tools: tools.into_iter().map(Into::into).collect(),
            blocked_tools: vec![],
        }
    }

    /// Create a scope that blocks the listed tools and allows everything else.
    pub fn blacklist(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        ScopeConfig {
            allowed_tools: vec![],
            blocked_tools: tools.into_iter().map(Into::into).collect(),
        }
    }

    /// Check if all requested tools are within scope.
    ///
    /// Returns the name of the first disallowed tool, or None if all are allowed.
    pub fn check_tools<'a>(&self, requested: &'a [String]) -> Option<&'a str> {
        for tool in requested {
            // Check explicit blocklist first.
            if self.blocked_tools.iter().any(|b| b == tool) {
                return Some(tool.as_str());
            }

            // Check whitelist (if configured).
            if !self.allowed_tools.is_empty()
                && !self.allowed_tools.iter().any(|a| a == tool)
            {
                return Some(tool.as_str());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_scope_allows_all() {
        let scope = ScopeConfig::default();
        assert!(scope.check_tools(&["any_tool".to_string()]).is_none());
    }

    #[test]
    fn test_whitelist_blocks_unknown() {
        let scope = ScopeConfig::whitelist(["get_order", "process_refund"]);
        assert!(scope.check_tools(&["get_order".to_string()]).is_none());
        assert!(scope.check_tools(&["delete_database".to_string()]).is_some());
    }

    #[test]
    fn test_blacklist_blocks_listed() {
        let scope = ScopeConfig::blacklist(["delete_database", "drop_table"]);
        assert!(scope.check_tools(&["get_order".to_string()]).is_none());
        assert!(scope.check_tools(&["delete_database".to_string()]).is_some());
    }
}
