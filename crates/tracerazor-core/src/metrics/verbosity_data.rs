/// Shared word lists for verbosity metrics (VDI, SHL, CCR).
///
/// All phrase lookups are applied to lowercased text via substring search.
/// Single-word FILLER_WORDS are matched against individual whitespace-split tokens.
/// Hedge phrases — indicate uncertain or weakly-qualified statements.
pub const HEDGE_PHRASES: &[&str] = &[
    "might",
    "could",
    "perhaps",
    "possibly",
    "i think",
    "it seems",
    "i believe",
    "seemingly",
    "probably",
    "it appears",
    "it would seem",
];

/// Preamble patterns — signal the agent is warming up before reasoning.
/// Matched as substring at the start of a sentence (or anywhere for count weighting).
pub const PREAMBLE_PATTERNS: &[&str] = &[
    "let me",
    "i need to",
    "i should",
    "i will now",
    "i'd be happy to",
    "i would be happy to",
    "certainly",
    "absolutely",
    "great question",
    "happy to help",
    "of course",
    "sure,",
    "i can help",
];

/// Single-word filler tokens that carry no semantic content.
pub const FILLER_WORDS: &[&str] = &[
    "basically",
    "actually",
    "essentially",
    "literally",
    "just",
    "really",
    "very",
    "furthermore",
    "additionally",
    "moreover",
    "needlessly",
    "obviously",
    "clearly",
    "simply",
    "quite",
    "rather",
    "somewhat",
];