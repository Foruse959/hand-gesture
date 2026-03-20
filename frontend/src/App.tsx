import { useEffect, useMemo, useState } from 'react'

import './App.css'
import { CameraOverlay } from './components/CameraOverlay.tsx'
import { LightweightGestureLab } from './components/LightweightGestureLab.tsx'
import { ProfileManagerPage } from './components/ProfileManagerPage'

type DashboardSnapshot = {
  sessions: number
  samples: number
  jobs: number
  average_accuracy: number
  best_latency_ms: number
}

type AppPage = 'studio' | 'profiles' | 'project' | 'about'

const PAGE_ITEMS: Array<{ id: AppPage; label: string; detail: string }> = [
  {
    id: 'studio',
    label: 'Studio',
    detail: 'Train and run profiles live.',
  },
  {
    id: 'profiles',
    label: 'Profiles',
    detail: 'Manage actions and gesture labels.',
  },
  {
    id: 'project',
    label: 'Project',
    detail: 'Architecture and capability highlights.',
  },
  {
    id: 'about',
    label: 'About',
    detail: 'Goals, direction, and API quick links.',
  },
]

const STUDIO_STEPS: Array<{ step: string; title: string; detail: string }> = [
  {
    step: '01',
    title: 'Open Studio',
    detail: 'Start with your profile and camera workflow in one clear vertical flow.',
  },
  {
    step: '02',
    title: 'Train Gestures',
    detail: 'Capture static and motion gestures, then inspect confidence before execution.',
  },
  {
    step: '03',
    title: 'Manage in Profiles',
    detail: 'Edit designed actions and labels on a dedicated management page.',
  },
  {
    step: '04',
    title: 'Start Live Profile',
    detail: 'Run your selected profile with cleaner spacing and reduced visual clutter.',
  },
]

const PROJECT_PILLARS: Array<{ title: string; detail: string; note: string }> = [
  {
    title: 'Adaptive Gesture Intelligence',
    detail: 'Two inference tracks: lightweight prototype matching and deep ResNet + LSTM support.',
    note: 'Optimized for custom labels and user-specific training.',
  },
  {
    title: 'Live Interaction Design',
    detail: 'Studio, Profiles, and camera panel cooperate without crowding each other.',
    note: 'Purposefully designed for usability, not only demos.',
  },
  {
    title: 'System Expansion Ready',
    detail: 'Desktop companion can execute mapped commands outside the browser when needed.',
    note: 'Web-first workflow with optional OS-level extension path.',
  },
]

const ABOUT_POINTS = [
  {
    label: 'Vision',
    title: 'Human-first interaction',
    detail: 'Enable practical gesture automation with confidence-aware behavior and clean workflows.',
  },
  {
    label: 'Engineering',
    title: 'Hybrid inference architecture',
    detail: 'Fast lightweight inference for responsiveness plus deep model support for harder classes.',
  },
  {
    label: 'Product',
    title: 'Profile-centric operations',
    detail: 'Profiles own labels, training state, and mapped actions so users can manage behavior cleanly.',
  },
]

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function App() {
  const [backendOnline, setBackendOnline] = useState(false)
  const [dashboard, setDashboard] = useState<DashboardSnapshot | null>(null)
  const [activePage, setActivePage] = useState<AppPage>('studio')
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    void bootstrap()

    const interval = window.setInterval(() => {
      void refreshDashboard()
    }, 12000)

    return () => {
      window.clearInterval(interval)
    }
  }, [])

  async function bootstrap() {
    try {
      const [healthRes, dashboardRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/dashboard'),
      ])

      setBackendOnline(healthRes.ok)

      if (dashboardRes.ok) {
        setDashboard((await dashboardRes.json()) as DashboardSnapshot)
      }
    } catch {
      setBackendOnline(false)
    }
  }

  async function refreshDashboard() {
    try {
      const [healthRes, dashboardRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/dashboard'),
      ])
      setBackendOnline(healthRes.ok)
      if (dashboardRes.ok) {
        setDashboard((await dashboardRes.json()) as DashboardSnapshot)
      }
    } catch {
      setBackendOnline(false)
    }
  }

  const currentPageMeta = useMemo(
    () => PAGE_ITEMS.find((item) => item.id === activePage) ?? PAGE_ITEMS[0],
    [activePage],
  )

  function handleSelectPage(page: AppPage) {
    setActivePage(page)
    setMenuOpen(false)
  }

  return (
    <div className="app-shell-v2">
      <header className="app-toolbar">
        <button
          type="button"
          className={`menu-toggle ${menuOpen ? 'open' : ''}`}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((value) => !value)}
        >
          <span />
          <span />
          <span />
        </button>

        <div className="toolbar-brand">
          <p className="eyebrow">Gesture Workspace</p>
          <h1>Dynamic Gesture Studio</h1>
          <p className="toolbar-page-caption">
            {currentPageMeta.label}: {currentPageMeta.detail}
          </p>
        </div>

        <div className="toolbar-status-strip">
          <span className={`status-pill status-main ${backendOnline ? 'online' : 'offline'}`}>
            {backendOnline ? 'Backend online' : 'Backend offline'}
          </span>
          <span className="status-pill status-subtle">
            {dashboard ? `${dashboard.samples} samples tracked` : 'No samples yet'}
          </span>
          <span className="status-pill status-subtle">
            {dashboard ? `${formatPercent(dashboard.average_accuracy)} avg accuracy` : '0.0% avg accuracy'}
          </span>
        </div>
      </header>

      <aside className={`page-drawer ${menuOpen ? 'open' : ''}`} aria-hidden={!menuOpen}>
        <p className="drawer-title">Navigation</p>
        <div className="drawer-nav-list">
          {PAGE_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`drawer-nav-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => handleSelectPage(item.id)}
            >
              <strong>{item.label}</strong>
              <span>{item.detail}</span>
            </button>
          ))}
        </div>

        <div className="drawer-mini-kpi">
          <div>
            <span>Jobs</span>
            <strong>{dashboard?.jobs ?? 0}</strong>
          </div>
          <div>
            <span>Sessions</span>
            <strong>{dashboard?.sessions ?? 0}</strong>
          </div>
          <div>
            <span>Best latency</span>
            <strong>{dashboard ? `${dashboard.best_latency_ms.toFixed(1)} ms` : '0.0 ms'}</strong>
          </div>
        </div>
      </aside>

      {menuOpen && (
        <button
          type="button"
          className="drawer-backdrop"
          aria-label="Close menu"
          onClick={() => setMenuOpen(false)}
        />
      )}

      <main className="page-main">
        {activePage === 'studio' && (
          <section className="page-shell">
            <section className="hero-panel-v2">
              <div>
                <p className="hero-kicker">Studio mode</p>
                <h2>Clear training flow with dedicated management pages.</h2>
                <p className="hero-summary">
                  This view now focuses on capture and live execution. Profile management moved to a separate page to keep camera and workflow unobstructed.
                </p>
              </div>
              <div className="hero-side-v2">
                <div className="kpi-list-v2">
                  <div>
                    <span>Sessions</span>
                    <strong>{dashboard?.sessions ?? 0}</strong>
                  </div>
                  <div>
                    <span>Samples</span>
                    <strong>{dashboard?.samples ?? 0}</strong>
                  </div>
                  <div>
                    <span>Training jobs</span>
                    <strong>{dashboard?.jobs ?? 0}</strong>
                  </div>
                </div>
              </div>
            </section>

            <section className="quick-steps-v2 glass-card">
              <div className="section-heading">
                <p className="eyebrow">Workflow</p>
                <h3>Use this order for best result</h3>
              </div>
              <div className="quick-steps-grid">
                {STUDIO_STEPS.map((item) => (
                  <article key={item.step} className="quick-step-card">
                    <span>Step {item.step}</span>
                    <h4>{item.title}</h4>
                    <p>{item.detail}</p>
                  </article>
                ))}
              </div>
            </section>

            <LightweightGestureLab backendOnline={backendOnline} />
          </section>
        )}

        {activePage === 'profiles' && (
          <ProfileManagerPage backendOnline={backendOnline} />
        )}

        {activePage === 'project' && (
          <section className="page-shell">
            <section className="glass-card">
              <div className="section-heading">
                <p className="eyebrow">Project</p>
                <h3>Platform highlights</h3>
              </div>
              <div className="showcase-grid project-page-shell">
                {PROJECT_PILLARS.map((item) => (
                  <article key={item.title} className="glass-card showcase-notes project-pillars-card">
                    <h4>{item.title}</h4>
                    <p className="muted-copy">{item.detail}</p>
                    <p className="pillar-note">{item.note}</p>
                  </article>
                ))}
              </div>
            </section>
          </section>
        )}

        {activePage === 'about' && (
          <section className="page-shell">
            <section className="glass-card help-shell about-shell">
              <div className="section-heading">
                <p className="eyebrow">About</p>
                <h3>Built for practical gesture interaction</h3>
              </div>
              <p className="muted-copy about-intro">
                Dynamic Gesture Studio is focused on reliable user-defined gestures, profile-level control, and clear management UX.
              </p>
              <div className="about-pillars-grid">
                {ABOUT_POINTS.map((item) => (
                  <article key={item.title} className="api-card about-card">
                    <span className="api-method">{item.label}</span>
                    <strong>{item.title}</strong>
                    <p>{item.detail}</p>
                  </article>
                ))}
              </div>

              <div className="about-api-block">
                <h4>API quick reference</h4>
                <div className="api-mini-grid">
                  <article className="api-card">
                    <span className="api-method">GET</span>
                    <strong>/api/light/profiles</strong>
                    <p>List stored profiles.</p>
                  </article>
                  <article className="api-card">
                    <span className="api-method">POST</span>
                    <strong>/api/light/predict</strong>
                    <p>Predict gesture from landmark sequence.</p>
                  </article>
                  <article className="api-card">
                    <span className="api-method">POST</span>
                    <strong>/api/light/execute</strong>
                    <p>Trigger mapped action by label and context.</p>
                  </article>
                </div>
              </div>
            </section>
          </section>
        )}
      </main>

      <CameraOverlay backendOnline={backendOnline} />
    </div>
  )
}

export default App
