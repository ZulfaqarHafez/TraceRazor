import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { fetchDashboard, fetchTraces, fetchTrace, deleteTrace, auditTrace, connectWs } from './api'

const GRADE_COLOR = { Excellent: '#22c55e', Good: '#3b82f6', Fair: '#f59e0b', Poor: '#ef4444' }

function StatCard({ label, value, sub }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={styles.cardValue}>{value}</div>
      {sub && <div style={styles.cardSub}>{sub}</div>}
    </div>
  )
}

function GradeBadge({ grade }) {
  return (
    <span style={{ ...styles.badge, background: GRADE_COLOR[grade] || '#64748b' }}>
      {grade}
    </span>
  )
}

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [dashboard, setDashboard] = useState(null)
  const [traces, setTraces] = useState([])
  const [selectedTrace, setSelectedTrace] = useState(null)
  const [liveEvents, setLiveEvents] = useState([])
  const [auditInput, setAuditInput] = useState('')
  const [auditResult, setAuditResult] = useState(null)
  const [auditError, setAuditError] = useState(null)
  const [loading, setLoading] = useState(false)

  const reload = useCallback(async () => {
    try {
      const [d, t] = await Promise.all([fetchDashboard(), fetchTraces()])
      setDashboard(d)
      setTraces(t)
    } catch (e) {
      console.error(e)
    }
  }, [])

  useEffect(() => {
    reload()
    const ws = connectWs((event) => {
      setLiveEvents((prev) => [event, ...prev].slice(0, 20))
      // Reload dashboard on new trace
      if (event.type === 'trace_analysed') reload()
    })
    return () => ws.close()
  }, [reload])

  async function handleSelectTrace(id) {
    try {
      const t = await fetchTrace(id)
      setSelectedTrace(t)
      setTab('detail')
    } catch (e) {
      console.error(e)
    }
  }

  async function handleDelete(id) {
    if (!confirm(`Delete trace ${id}?`)) return
    await deleteTrace(id)
    reload()
    if (selectedTrace?.trace?.trace_id === id) {
      setSelectedTrace(null)
      setTab('traces')
    }
  }

  async function handleAudit() {
    setAuditResult(null)
    setAuditError(null)
    setLoading(true)
    try {
      const trace = JSON.parse(auditInput)
      const result = await auditTrace(trace)
      setAuditResult(result)
      reload()
    } catch (e) {
      setAuditError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <span style={styles.logo}>⚡ TraceRazor</span>
        <nav style={styles.nav}>
          {['dashboard', 'traces', 'audit', 'live'].map((t) => (
            <button key={t} style={tab === t ? styles.tabActive : styles.tab}
              onClick={() => setTab(t)}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        {liveEvents.length > 0 && (
          <span style={styles.liveDot} title="Live events streaming">● LIVE</span>
        )}
      </header>

      <main style={styles.main}>
        {tab === 'dashboard' && dashboard && (
          <DashboardTab dashboard={dashboard} onSelectTrace={handleSelectTrace} />
        )}
        {tab === 'traces' && (
          <TracesTab traces={traces} onSelect={handleSelectTrace} onDelete={handleDelete} />
        )}
        {tab === 'audit' && (
          <AuditTab
            input={auditInput} onInput={setAuditInput}
            onAudit={handleAudit} loading={loading}
            result={auditResult} error={auditError}
          />
        )}
        {tab === 'live' && <LiveTab events={liveEvents} />}
        {tab === 'detail' && selectedTrace && (
          <DetailTab stored={selectedTrace} onBack={() => setTab('traces')} onDelete={handleDelete} />
        )}
        {tab === 'dashboard' && !dashboard && (
          <div style={styles.empty}>Loading dashboard…</div>
        )}
      </main>
    </div>
  )
}

// ── Dashboard Tab ─────────────────────────────────────────────────────────────
function DashboardTab({ dashboard, onSelectTrace }) {
  const trend = dashboard.tas_trend?.slice(-30) ?? []
  return (
    <div>
      <h2 style={styles.sectionTitle}>Overview</h2>
      <div style={styles.statRow}>
        <StatCard label="Total Traces" value={dashboard.total_traces} />
        <StatCard label="Active Agents" value={dashboard.total_agents} />
        <StatCard label="Avg TAS Score" value={`${dashboard.avg_tas?.toFixed(1) ?? '—'} / 100`} />
        <StatCard
          label="Tokens Saved"
          value={Number(dashboard.total_tokens_saved).toLocaleString()}
          sub={`$${dashboard.total_cost_saved_usd?.toFixed(2)} saved`}
        />
      </div>

      {trend.length > 0 && (
        <>
          <h2 style={styles.sectionTitle}>TAS Score Trend</h2>
          <div style={styles.chartBox}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="timestamp" tickFormatter={(t) => t.slice(0, 10)} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 8 }}
                  labelFormatter={(l) => l?.slice(0, 10)}
                />
                <Legend />
                <Line type="monotone" dataKey="tas_score" stroke="#3b82f6" dot={false} strokeWidth={2} name="TAS Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {dashboard.agent_rankings?.length > 0 && (
        <>
          <h2 style={styles.sectionTitle}>Agent Rankings (worst first)</h2>
          <table style={styles.table}>
            <thead>
              <tr>
                {['Agent', 'Traces', 'Avg TAS', 'Min', 'Max', 'Tokens Saved'].map((h) => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dashboard.agent_rankings.map((a) => (
                <tr key={a.agent_name}>
                  <td style={styles.td}>{a.agent_name}</td>
                  <td style={styles.td}>{a.trace_count}</td>
                  <td style={styles.td}>{a.avg_tas?.toFixed(1)}</td>
                  <td style={styles.td}>{a.min_tas?.toFixed(1)}</td>
                  <td style={styles.td}>{a.max_tas?.toFixed(1)}</td>
                  <td style={styles.td}>{Number(a.total_tokens_saved).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {dashboard.recent_traces?.length > 0 && (
        <>
          <h2 style={styles.sectionTitle}>Recent Traces</h2>
          <TracesTab traces={dashboard.recent_traces} onSelect={onSelectTrace} />
        </>
      )}
    </div>
  )
}

// ── Traces Tab ────────────────────────────────────────────────────────────────
function TracesTab({ traces, onSelect, onDelete }) {
  if (!traces?.length) return <div style={styles.empty}>No traces yet.</div>
  return (
    <table style={styles.table}>
      <thead>
        <tr>
          {['Trace ID', 'Agent', 'Framework', 'Steps', 'Tokens', 'TAS', 'Grade', 'Stored', ''].map((h) => (
            <th key={h} style={styles.th}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {traces.map((t) => (
          <tr key={t.trace_id} style={styles.trHover} onClick={() => onSelect?.(t.trace_id)}>
            <td style={{ ...styles.td, fontFamily: 'monospace', fontSize: 12 }}>{t.trace_id}</td>
            <td style={styles.td}>{t.agent_name}</td>
            <td style={styles.td}>{t.framework}</td>
            <td style={styles.td}>{t.total_steps}</td>
            <td style={styles.td}>{Number(t.total_tokens).toLocaleString()}</td>
            <td style={styles.td}>{t.tas_score?.toFixed(1) ?? '—'}</td>
            <td style={styles.td}>{t.grade ? <GradeBadge grade={t.grade} /> : '—'}</td>
            <td style={styles.td}>{t.stored_at?.slice(0, 10)}</td>
            <td style={styles.td} onClick={(e) => { e.stopPropagation(); onDelete?.(t.trace_id) }}>
              {onDelete && <button style={styles.deleteBtn}>✕</button>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Audit Tab ─────────────────────────────────────────────────────────────────
function AuditTab({ input, onInput, onAudit, loading, result, error }) {
  return (
    <div style={{ maxWidth: 800 }}>
      <h2 style={styles.sectionTitle}>Audit a Trace</h2>
      <p style={{ color: '#94a3b8', marginTop: 0 }}>
        Paste raw trace JSON below and click Analyse.
      </p>
      <textarea
        style={styles.textarea}
        value={input}
        onChange={(e) => onInput(e.target.value)}
        placeholder='{"trace_id": "...", "agent_name": "...", ...}'
        rows={14}
      />
      <button style={styles.button} onClick={onAudit} disabled={loading || !input.trim()}>
        {loading ? 'Analysing…' : 'Analyse Trace'}
      </button>
      {error && <div style={styles.errorBox}>{error}</div>}
      {result && (
        <div style={styles.resultBox}>
          <h3 style={{ marginTop: 0 }}>
            {result.agent_name} — <GradeBadge grade={result.grade} /> {result.tas_score?.toFixed(1)} / 100
          </h3>
          <p>Tokens saved: <strong>{Number(result.tokens_saved).toLocaleString()}</strong></p>
          <pre style={styles.pre}>{result.report_markdown}</pre>
        </div>
      )}
    </div>
  )
}

// ── Live Events Tab ───────────────────────────────────────────────────────────
function LiveTab({ events }) {
  if (!events.length) return (
    <div style={styles.empty}>
      Waiting for live events… Connect an agent and audit traces to see real-time updates.
    </div>
  )
  return (
    <div>
      <h2 style={styles.sectionTitle}>Live Events</h2>
      {events.map((e, i) => (
        <div key={i} style={styles.eventCard}>
          <span style={styles.eventType}>{e.type}</span>
          {e.type === 'trace_analysed' && (
            <span> — {e.agent_name} scored <strong>{e.tas_score?.toFixed(1)}</strong>
              &nbsp;<GradeBadge grade={e.grade} />, saved {e.tokens_saved} tokens
            </span>
          )}
          {e.type === 'loop_detected' && (
            <span> — Loop at step {e.step_id}: {e.cycle}</span>
          )}
        </div>
      ))}
    </div>
  )
}

// ── Detail Tab ────────────────────────────────────────────────────────────────
function DetailTab({ stored, onBack, onDelete }) {
  const { trace, report } = stored
  return (
    <div>
      <button style={styles.backBtn} onClick={onBack}>← Back</button>
      <h2 style={styles.sectionTitle}>
        {trace.trace_id}
        <button style={{ ...styles.deleteBtn, marginLeft: 12 }}
          onClick={() => onDelete(trace.trace_id)}>Delete</button>
      </h2>
      <div style={styles.statRow}>
        <StatCard label="Agent" value={trace.agent_name} />
        <StatCard label="Framework" value={trace.framework} />
        <StatCard label="Steps" value={trace.steps?.length} />
        <StatCard label="Total Tokens" value={Number(trace.total_tokens).toLocaleString()} />
        {report && <>
          <StatCard label="TAS Score" value={`${report.score?.score?.toFixed(1)} / 100`} />
          <StatCard label="Grade" value={<GradeBadge grade={report.score?.grade} />} />
        </>}
      </div>
      {report && (
        <pre style={{ ...styles.pre, marginTop: 24 }}>{report.to_markdown?.() ?? JSON.stringify(report, null, 2)}</pre>
      )}
      {!report && (
        <pre style={styles.pre}>{JSON.stringify(trace, null, 2)}</pre>
      )}
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  root: { minHeight: '100vh', display: 'flex', flexDirection: 'column' },
  header: {
    display: 'flex', alignItems: 'center', gap: 16,
    padding: '12px 24px', background: '#0a0e1a',
    borderBottom: '1px solid #1e293b',
  },
  logo: { fontWeight: 700, fontSize: 18, color: '#3b82f6', marginRight: 16 },
  nav: { display: 'flex', gap: 4, flex: 1 },
  tab: {
    background: 'none', border: 'none', color: '#94a3b8',
    cursor: 'pointer', padding: '6px 14px', borderRadius: 6, fontSize: 14,
  },
  tabActive: {
    background: '#1e293b', border: 'none', color: '#e2e8f0',
    cursor: 'pointer', padding: '6px 14px', borderRadius: 6, fontSize: 14, fontWeight: 600,
  },
  liveDot: { color: '#22c55e', fontSize: 12, fontWeight: 700 },
  main: { flex: 1, padding: '24px 32px', maxWidth: 1200, width: '100%', margin: '0 auto' },
  sectionTitle: { color: '#e2e8f0', fontSize: 16, fontWeight: 600, marginBottom: 12 },
  statRow: { display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 },
  card: {
    background: '#1e293b', borderRadius: 10, padding: '16px 20px',
    minWidth: 160, flex: '1 1 160px',
  },
  cardLabel: { color: '#64748b', fontSize: 12, marginBottom: 4 },
  cardValue: { color: '#e2e8f0', fontSize: 22, fontWeight: 700 },
  cardSub: { color: '#22c55e', fontSize: 12, marginTop: 4 },
  chartBox: { background: '#1e293b', borderRadius: 10, padding: '16px', marginBottom: 24 },
  table: { width: '100%', borderCollapse: 'collapse', marginBottom: 24 },
  th: {
    textAlign: 'left', padding: '10px 12px', fontSize: 12, color: '#64748b',
    borderBottom: '1px solid #1e293b',
  },
  td: { padding: '10px 12px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #0f1117' },
  trHover: { cursor: 'pointer' },
  badge: {
    display: 'inline-block', borderRadius: 4, padding: '2px 8px',
    fontSize: 11, fontWeight: 700, color: '#fff',
  },
  deleteBtn: {
    background: 'none', border: 'none', color: '#ef4444',
    cursor: 'pointer', fontSize: 14, padding: '2px 6px',
  },
  backBtn: {
    background: '#1e293b', border: 'none', color: '#94a3b8',
    cursor: 'pointer', padding: '6px 14px', borderRadius: 6, fontSize: 14, marginBottom: 16,
  },
  textarea: {
    width: '100%', background: '#1e293b', border: '1px solid #334155',
    borderRadius: 8, color: '#e2e8f0', padding: 12, fontSize: 13, fontFamily: 'monospace',
    resize: 'vertical',
  },
  button: {
    background: '#3b82f6', border: 'none', color: '#fff',
    padding: '10px 20px', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 600,
    marginTop: 12,
  },
  errorBox: {
    marginTop: 12, background: '#450a0a', border: '1px solid #ef4444',
    borderRadius: 8, padding: 12, color: '#fca5a5', fontSize: 13,
  },
  resultBox: {
    marginTop: 16, background: '#0f2d1f', border: '1px solid #22c55e',
    borderRadius: 8, padding: 16,
  },
  pre: {
    background: '#0f1117', borderRadius: 8, padding: 16,
    fontSize: 12, color: '#94a3b8', overflow: 'auto', whiteSpace: 'pre-wrap',
  },
  eventCard: {
    background: '#1e293b', borderRadius: 8, padding: '10px 14px', marginBottom: 8, fontSize: 13,
  },
  eventType: { fontWeight: 700, color: '#3b82f6', fontFamily: 'monospace' },
  empty: { color: '#64748b', textAlign: 'center', padding: '48px 0' },
}
