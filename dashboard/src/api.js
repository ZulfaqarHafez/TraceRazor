const BASE = '/api'

export async function fetchDashboard() {
  const r = await fetch(`${BASE}/dashboard`)
  if (!r.ok) throw new Error(`Dashboard fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchTraces() {
  const r = await fetch(`${BASE}/traces`)
  if (!r.ok) throw new Error(`Traces fetch failed: ${r.status}`)
  return r.json()
}

export async function fetchTrace(id) {
  const r = await fetch(`${BASE}/traces/${id}`)
  if (!r.ok) throw new Error(`Trace fetch failed: ${r.status}`)
  return r.json()
}

export async function deleteTrace(id) {
  const r = await fetch(`${BASE}/traces/${id}`, { method: 'DELETE' })
  if (!r.ok && r.status !== 204) throw new Error(`Delete failed: ${r.status}`)
}

export async function auditTrace(traceJson) {
  const r = await fetch(`${BASE}/audit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace: traceJson }),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }))
    throw new Error(err.error || 'Audit failed')
  }
  return r.json()
}

/** Connect to the WebSocket and call onEvent(event) for each message. */
export function connectWs(onEvent) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws`)
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)) } catch (_) {}
  }
  ws.onerror = (e) => console.warn('WS error', e)
  return ws
}
