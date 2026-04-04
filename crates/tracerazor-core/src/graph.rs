use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

use crate::types::{StepType, Trace, TraceStep};

/// A node in the trace DAG.
#[derive(Debug, Clone)]
pub struct TraceNode {
    pub step_index: usize,
    pub agent_id: String,
}

/// Edge type in the trace DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    /// Sequential flow between steps of the same agent.
    Sequential,
    /// Handoff from one agent to another.
    Handoff,
}

/// Directed acyclic graph representation of a trace.
/// Each node corresponds to a step; edges represent execution flow.
pub struct TraceGraph {
    pub graph: DiGraph<TraceNode, EdgeKind>,
    /// Map from step id (1-based) to NodeIndex
    pub node_map: HashMap<u32, NodeIndex>,
}

impl TraceGraph {
    /// Build a TraceGraph from a parsed Trace.
    ///
    /// For single-agent traces, steps are connected sequentially.
    /// For multi-agent traces, each agent's steps are connected sequentially within
    /// their thread, with cross-thread (handoff) edges where agents interact.
    pub fn from_trace(trace: &Trace) -> Self {
        let mut graph: DiGraph<TraceNode, EdgeKind> = DiGraph::new();
        let mut node_map: HashMap<u32, NodeIndex> = HashMap::new();

        // Add all steps as nodes.
        for step in &trace.steps {
            let node = TraceNode {
                step_index: (step.id - 1) as usize,
                agent_id: step.agent_id.clone().unwrap_or_default(),
            };
            let idx = graph.add_node(node);
            node_map.insert(step.id, idx);
        }

        // Add sequential edges between consecutive steps.
        // For multi-agent traces, connect within the same agent thread and add
        // cross-thread edges for handoffs.
        let steps = &trace.steps;
        for i in 1..steps.len() {
            let prev = &steps[i - 1];
            let curr = &steps[i];

            let prev_idx = node_map[&prev.id];
            let curr_idx = node_map[&curr.id];

            let same_agent = prev.agent_id == curr.agent_id;
            let edge_kind = if same_agent || prev.step_type == StepType::Handoff {
                EdgeKind::Sequential
            } else {
                EdgeKind::Handoff
            };

            graph.add_edge(prev_idx, curr_idx, edge_kind);
        }

        TraceGraph { graph, node_map }
    }

    /// Returns all node indices in topological order.
    pub fn topological_order(&self) -> Vec<NodeIndex> {
        match petgraph::algo::toposort(&self.graph, None) {
            Ok(order) => order,
            // If there are cycles (shouldn't happen in a valid trace but tolerate it),
            // fall back to insertion order.
            Err(_) => self.graph.node_indices().collect(),
        }
    }

    /// Detect all simple cycles in the graph.
    /// Returns a list of cycles, each cycle being a list of step IDs.
    pub fn detect_cycles(&self) -> Vec<Vec<u32>> {
        // Build reverse map: NodeIndex -> step_id
        let reverse_map: HashMap<NodeIndex, u32> = self
            .node_map
            .iter()
            .map(|(k, v)| (*v, *k))
            .collect();

        let sccs = petgraph::algo::kosaraju_scc(&self.graph);
        let mut cycles: Vec<Vec<u32>> = Vec::new();

        for scc in sccs {
            if scc.len() > 1 {
                // This SCC forms a cycle.
                let mut step_ids: Vec<u32> = scc
                    .iter()
                    .filter_map(|idx| reverse_map.get(idx))
                    .copied()
                    .collect();
                step_ids.sort();
                cycles.push(step_ids);
            }
        }

        cycles
    }

    /// Find repeated state sequences using state hashing.
    /// Returns pairs of (step_id_a, step_id_b) where the same state recurs.
    pub fn find_repeated_states(steps: &[TraceStep]) -> Vec<(u32, u32)> {
        let mut seen: HashMap<String, u32> = HashMap::new();
        let mut repeats: Vec<(u32, u32)> = Vec::new();

        for step in steps {
            let hash = step.state_hash();
            // Only match non-trivial hashes (tool calls with actual names).
            if step.tool_name.is_some() {
                if let Some(&prev_id) = seen.get(&hash) {
                    repeats.push((prev_id, step.id));
                } else {
                    seen.insert(hash, step.id);
                }
            }
        }

        repeats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StepType, TraceStep};

    fn make_step(id: u32, step_type: StepType, tokens: u32) -> TraceStep {
        TraceStep {
            id,
            step_type,
            content: format!("Step {id} content"),
            tokens,
            tool_name: None,
            tool_params: None,
            tool_success: None,
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    fn make_tool_step(id: u32, tool: &str, tokens: u32) -> TraceStep {
        TraceStep {
            id,
            step_type: StepType::ToolCall,
            content: format!("Calling {tool}"),
            tokens,
            tool_name: Some(tool.to_string()),
            tool_params: Some(serde_json::json!({"param": "value"})),
            tool_success: Some(true),
            tool_error: None,
            agent_id: None,
            input_context: None,
            output: None,
            flags: vec![],
            flag_details: vec![],
        }
    }

    #[test]
    fn test_graph_construction() {
        let trace = Trace {
            trace_id: "t1".into(),
            agent_name: "agent".into(),
            framework: "raw".into(),
            steps: vec![
                make_step(1, StepType::Reasoning, 100),
                make_step(2, StepType::ToolCall, 200),
                make_step(3, StepType::Reasoning, 150),
            ],
            total_tokens: 450,
            task_value_score: 1.0,
            metadata: Default::default(),
        };

        let tg = TraceGraph::from_trace(&trace);
        assert_eq!(tg.graph.node_count(), 3);
        assert_eq!(tg.graph.edge_count(), 2);
    }

    #[test]
    fn test_repeated_state_detection() {
        let steps = vec![
            make_tool_step(1, "get_order", 100),
            make_step(2, StepType::Reasoning, 200),
            make_tool_step(3, "get_order", 100), // same as step 1
        ];
        let repeats = TraceGraph::find_repeated_states(&steps);
        assert_eq!(repeats.len(), 1);
        assert_eq!(repeats[0], (1, 3));
    }
}
