function app() {
  return {
    dark: localStorage.getItem('tr-theme') !== 'light',
    tab: 'dashboard',
    tabs: [
      {id:'dashboard', label:'Dashboard'},
      {id:'traces',    label:'Traces'},
      {id:'audit',     label:'Audit'},
      {id:'compare',   label:'Compare'},
      {id:'kb',        label:'KB'},
      {id:'live',      label:'Live'},
    ],
    dashboard: null,
    traces: [],
    kbEntries: [],
    kbSelected: null,
    selectedTrace: null,
    liveEvents: [],
    auditInput: '',
    auditResult: null,
    auditError: null,
    auditLoading: false,
    compareA: '',
    compareB: '',
    compareResult: null,
    compareError: null,
    compareLoading: false,
    _chart: null,
    _ws: null,

    async boot() {
      await this.reload()
      this.connectWs()
    },

    toggleTheme() {
      this.dark = !this.dark
      localStorage.setItem('tr-theme', this.dark ? 'dark' : 'light')
      this.$nextTick(() => this.drawChart())
    },

    async reload() {
      try {
        const [d, t, kb] = await Promise.all([
          fetch('/api/dashboard').then(r => r.json()),
          fetch('/api/traces').then(r => r.json()),
          fetch('/api/kb').then(r => r.json()),
        ])
        this.dashboard = d
        this.traces = t
        this.kbEntries = kb
        this.$nextTick(() => this.drawChart())
      } catch(e) { console.error(e) }
    },

    drawChart() {
      const trend = this.dashboard?.tas_trend
      if (!trend || trend.length === 0) return
      const canvas = document.getElementById('trendChart')
      if (!canvas) return
      if (this._chart) { this._chart.destroy(); this._chart = null }
      const isDark = this.dark
      const gridColor  = isDark ? '#1e293b' : '#e2e8f0'
      const tickColor  = isDark ? '#64748b' : '#94a3b8'
      this._chart = new Chart(canvas, {
        type: 'line',
        data: {
          labels: trend.map(p => (p.timestamp || '').slice(0,10)),
          datasets: [{
            label: 'TAS Score',
            data: trend.map(p => p.tas_score),
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,0.08)',
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.3,
            fill: true,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: gridColor }, ticks: { color: tickColor, maxTicksLimit: 10 } },
            y: { min: 0, max: 100, grid: { color: gridColor }, ticks: { color: tickColor } },
          }
        }
      })
    },

    async openTrace(id) {
      try {
        this.selectedTrace = await fetch(`/api/traces/${id}`).then(r => r.json())
        this.tab = 'detail'
      } catch(e) { alert(e.message) }
    },

    async deleteKb(id) {
      if (!confirm(`Remove KB entry ${id}?`)) return
      await fetch(`/api/kb/${id}`, {method:'DELETE'})
      if (this.kbSelected?.id === id) this.kbSelected = null
      await this.reload()
    },

    async doDelete(id) {
      if (!confirm(`Delete trace ${id}?`)) return
      await fetch(`/api/traces/${id}`, {method:'DELETE'})
      if (this.selectedTrace?.trace?.trace_id === id) { this.selectedTrace = null; this.tab = 'traces' }
      await this.reload()
    },

    async doAudit() {
      this.auditResult = null; this.auditError = null; this.auditLoading = true
      try {
        const trace = JSON.parse(this.auditInput)
        const r = await fetch('/api/audit', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({trace}),
        })
        if (!r.ok) { const e = await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error) }
        this.auditResult = await r.json()
        await this.reload()
      } catch(e) { this.auditError = e.message }
      finally { this.auditLoading = false }
    },

    async doCompare() {
      this.compareResult = null; this.compareError = null; this.compareLoading = true
      try {
        const r = await fetch(`/api/compare?a=${encodeURIComponent(this.compareA)}&b=${encodeURIComponent(this.compareB)}`)
        if (!r.ok) { const e = await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error) }
        this.compareResult = await r.json()
      } catch(e) { this.compareError = e.message }
      finally { this.compareLoading = false }
    },

    connectWs() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onmessage = (e) => {
        try {
          const ev = JSON.parse(e.data)
          this.liveEvents = [ev, ...this.liveEvents].slice(0, 30)
          if (ev.type === 'trace_analysed') this.reload()
        } catch(_) {}
      }
      ws.onclose = () => setTimeout(() => this.connectWs(), 3000) // auto-reconnect
      this._ws = ws
    },

    // Template helpers: Alpine's CSP build cannot reach JS globals (JSON, Math)
    // from attribute expressions, so these live on the component.
    selectedTraceJson() {
      return JSON.stringify(this.selectedTrace, null, 2)
    },

    // The stored trace's own total_tokens is often 0 (token counts live on the
    // steps); prefer the audited report's total.
    selectedTraceTokens() {
      const t = this.selectedTrace
      if (!t) return 0
      return (t.report && t.report.total_tokens) || (t.trace && t.trace.total_tokens) || 0
    },

    kbReductionLabel() {
      const kb = this.kbSelected
      if (!kb || !kb.total_tokens) return ''
      return Math.round((1 - kb.optimal_tokens / kb.total_tokens) * 100) + '% reduction'
    },

    fmtNum: n => Number(n||0).toLocaleString(),
  }
}

// The CSP build of Alpine cannot evaluate `x-data="app()"`, so register the
// component by name; the page uses `x-data="app"`.
document.addEventListener('alpine:init', () => {
  Alpine.data('app', app)
})
