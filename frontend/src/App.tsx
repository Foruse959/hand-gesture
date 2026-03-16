import { useEffect, useState } from 'react'

import './App.css'
import { CameraOverlay } from './components/CameraOverlay.tsx'
import { LightweightGestureLab } from './components/LightweightGestureLab.tsx'

type DashboardSnapshot = {
  sessions: number
  samples: number
  jobs: number
  average_accuracy: number
  best_latency_ms: number
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function App() {
  const [backendOnline, setBackendOnline] = useState(false)
  const [dashboard, setDashboard] = useState<DashboardSnapshot | null>(null)

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

  return (
    <div className="app-shell app-shell-pro">
      <header className="site-header product-header">
        <div>
          <p className="eyebrow">Gesture Control Platform</p>
          <div className="brand-row">
            <h1>Dynamic Gesture Studio</h1>
            <span className={`status-pill ${backendOnline ? 'online' : 'offline'}`}>
              {backendOnline ? 'Backend online' : 'Backend offline'}
            </span>
          </div>
          <p className="product-subtitle">
            Train your own gestures, map real actions, and run a live hand-controlled workflow.
          </p>
        </div>

        <nav className="top-nav">
          <a href="#quick-start">Quick start</a>
          <a href="#lightweight-lab">Live lab</a>
          <a href="#showcase">Pointer demo</a>
          <a href="#help">Help</a>
        </nav>
      </header>

      <main className="main-layout product-layout">
        <section className="hero-panel product-hero">
          <div className="hero-copy">
            <p className="hero-kicker">Professional mode: practical, not documentation-heavy</p>
            <h2>
              Build your own hand gesture profile and use it for browser actions, typing, and live control.
            </h2>
            <p className="hero-summary">
              The app now focuses on operation first: create profile, train quickly, and go live. You can hide or dock
              the camera side panel, view tracking points, and use a virtual pointer while testing.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#lightweight-lab">
                Start now
              </a>
              <a className="secondary-button" href="#quick-start">
                2-minute setup
              </a>
            </div>
          </div>

          <div className="hero-side">
            <article className="glass-card emphasis-card quick-card">
              <span className="mini-label">Live status</span>
              <div className="kpi-list">
                <div>
                  <span>Profiles/sessions</span>
                  <strong>{dashboard?.sessions ?? 0}</strong>
                </div>
                <div>
                  <span>Captured samples</span>
                  <strong>{dashboard?.samples ?? 0}</strong>
                </div>
                <div>
                  <span>Training jobs</span>
                  <strong>{dashboard?.jobs ?? 0}</strong>
                </div>
                <div>
                  <span>Avg accuracy</span>
                  <strong>{dashboard ? formatPercent(dashboard.average_accuracy) : '0.0%'}</strong>
                </div>
                <div>
                  <span>Best latency</span>
                  <strong>{dashboard ? `${dashboard.best_latency_ms.toFixed(1)} ms` : '0.0 ms'}</strong>
                </div>
              </div>
            </article>
          </div>
        </section>

        <section id="quick-start" className="glass-card quick-start-shell">
          <div className="section-heading">
            <p className="eyebrow">Quick start</p>
            <h3>Do these 4 steps in order</h3>
          </div>

          <div className="quick-steps-grid">
            <article className="quick-step-card">
              <span>Step 1</span>
              <h4>Create profile</h4>
              <p>
                Enter gesture labels in the Profile card and click <strong>Create profile</strong>.
              </p>
            </article>
            <article className="quick-step-card">
              <span>Step 2</span>
              <h4>Start camera + train</h4>
              <p>
                Start webcam, keep hand visible, then click <strong>Capture + train</strong> for each gesture 4-8 times.
              </p>
            </article>
            <article className="quick-step-card">
              <span>Step 3</span>
              <h4>Map actions</h4>
              <p>
                In Action mapping, connect gestures to URL/app/hotkey actions and save.
              </p>
            </article>
            <article className="quick-step-card">
              <span>Step 4</span>
              <h4>Go live</h4>
              <p>
                Enable <strong>Live recognition mode</strong>. If needed, enable auto execute.
              </p>
            </article>
          </div>
        </section>

        <section className="glass-card beginner-notes-shell">
          <div className="section-heading">
            <p className="eyebrow">Beginner notes</p>
            <h3>What "profile" and "session" mean</h3>
          </div>
          <div className="quick-faq-grid">
            <article>
              <h4>Profile</h4>
              <p>
                Your personal gesture model. It stores labels and learned gesture patterns.
              </p>
            </article>
            <article>
              <h4>Session</h4>
              <p>
                Backend run history for dataset/training benchmarking. Optional for basic live control.
              </p>
            </article>
            <article>
              <h4>Where is the pointer?</h4>
              <p>
                Open the floating camera panel. Start camera and turn on pointer mode. A virtual pointer follows your
                index fingertip.
              </p>
            </article>
          </div>
        </section>

        <LightweightGestureLab backendOnline={backendOnline} />

        <section id="showcase" className="showcase-shell">
          <div className="section-heading">
            <p className="eyebrow">Pointer demo app</p>
            <h3>Try hand pointer interactions in the embedded legacy module</h3>
          </div>

          <div className="showcase-grid">
            <article className="glass-card showcase-notes">
              <h4>Use this for demo day</h4>
              <ul className="feature-list">
                <li>Open camera overlay and enable pointer mode.</li>
                <li>Use this page to show visible hand-tracking and cursor-like control.</li>
                <li>Use Live Lab for training and action mapping, then show this module for wow factor.</li>
              </ul>
            </article>

            <article className="iframe-shell">
              <iframe title="Legacy Gesture Shop" src="/legacy/gesture-shop.html" />
            </article>
          </div>
        </section>

        <section id="help" className="glass-card help-shell">
          <div className="section-heading">
            <p className="eyebrow">Use outside webpage</p>
            <h3>Profiles can be reused through API</h3>
          </div>
          <p className="muted-copy">
            Your trained profile is stored on backend. You can call API endpoints from another app/script to run
            prediction or execute mapped actions.
          </p>
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
              <p>Trigger mapped action by label/context.</p>
            </article>
          </div>
        </section>
      </main>

      <CameraOverlay backendOnline={backendOnline} />
    </div>
  )
}

export default App
