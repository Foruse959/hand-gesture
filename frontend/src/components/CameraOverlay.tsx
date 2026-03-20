import { useEffect, useRef, useState } from 'react'

import { useCameraFeed } from '../hooks/useCameraFeed'

type CameraOverlayProps = {
  backendOnline: boolean
}

type DockSide = 'left' | 'right'

declare global {
  interface Window {
    Hands?: any
  }
}

const HANDS_SCRIPT = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js'
const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]
const POINTER_SMOOTHING = 0.52

async function loadScript(url: string) {
  if (document.querySelector(`script[src="${url}"]`)) {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = url
    script.async = true
    script.crossOrigin = 'anonymous'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error(`Failed to load script: ${url}`))
    document.head.appendChild(script)
  })
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value))
}

export function CameraOverlay({ backendOnline }: CameraOverlayProps) {
  const {
    videoRef,
    state,
    error,
    isStreaming,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshDevices,
    start,
    stop,
  } = useCameraFeed()

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const handsRef = useRef<any>(null)
  const loopRef = useRef<number | null>(null)
  const processingRef = useRef(false)
  const mountedRef = useRef(true)

  const [minimized, setMinimized] = useState(false)
  const [hidden, setHidden] = useState(true)
  const [dockSide, setDockSide] = useState<DockSide>('right')
  const [pointerMode, setPointerMode] = useState(true)
  const [handDetected, setHandDetected] = useState(false)
  const [landmarkCount, setLandmarkCount] = useState(0)
  const [pointer, setPointer] = useState({ x: 0.5, y: 0.5 })

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setHidden(true)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  function clearCanvas() {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const context = canvas.getContext('2d')
    if (!context) {
      return
    }
    context.clearRect(0, 0, canvas.width, canvas.height)
  }

  function drawLandmarks(points: Array<{ x: number; y: number }> | null) {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) {
      return
    }

    const rect = video.getBoundingClientRect()
    if (rect.width < 2 || rect.height < 2) {
      return
    }

    const ratio = window.devicePixelRatio || 1
    const targetWidth = Math.floor(rect.width * ratio)
    const targetHeight = Math.floor(rect.height * ratio)
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth
      canvas.height = targetHeight
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.scale(ratio, ratio)

    if (!points) {
      return
    }

    context.strokeStyle = 'rgba(214, 255, 87, 0.95)'
    context.lineWidth = 2

    HAND_CONNECTIONS.forEach(([startIndex, endIndex]) => {
      const startPoint = points[startIndex]
      const endPoint = points[endIndex]
      if (!startPoint || !endPoint) {
        return
      }
      context.beginPath()
      context.moveTo((1 - startPoint.x) * rect.width, startPoint.y * rect.height)
      context.lineTo((1 - endPoint.x) * rect.width, endPoint.y * rect.height)
      context.stroke()
    })

    context.fillStyle = 'rgba(127, 209, 255, 0.98)'
    points.forEach((point, index) => {
      const radius = index === 8 ? 6 : 4
      context.beginPath()
      context.arc((1 - point.x) * rect.width, point.y * rect.height, radius, 0, Math.PI * 2)
      context.fill()
    })
  }

  useEffect(() => {
    if (!isStreaming) {
      if (loopRef.current) {
        window.cancelAnimationFrame(loopRef.current)
        loopRef.current = null
      }
      handsRef.current = null
      processingRef.current = false
      setHandDetected(false)
      setLandmarkCount(0)
      clearCanvas()
      return
    }

    let cancelled = false

    const startTracking = async () => {
      try {
        await loadScript(HANDS_SCRIPT)
        const video = videoRef.current
        if (!video || !window.Hands || cancelled) {
          return
        }

        const hands = new window.Hands({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        })

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 0,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.4,
        })

        hands.onResults((results: any) => {
          const detected = results?.multiHandLandmarks?.[0] as Array<{ x: number; y: number }> | undefined
          if (!mountedRef.current || cancelled) {
            return
          }

          if (!detected) {
            setHandDetected(false)
            setLandmarkCount(0)
            drawLandmarks(null)
            return
          }

          setHandDetected(true)
          setLandmarkCount(detected.length)

          const tip = detected[8]
          if (tip) {
            const targetX = clamp01(1 - tip.x)
            const targetY = clamp01(tip.y)
            setPointer((prev) => ({
              x: prev.x + (targetX - prev.x) * POINTER_SMOOTHING,
              y: prev.y + (targetY - prev.y) * POINTER_SMOOTHING,
            }))
          }

          drawLandmarks(detected)
        })

        handsRef.current = hands

        const loop = async () => {
          const runningVideo = videoRef.current
          const runningHands = handsRef.current
          if (cancelled || !runningVideo || !runningHands) {
            return
          }

          if (!processingRef.current && runningVideo.readyState >= 2) {
            processingRef.current = true
            try {
              await runningHands.send({ image: runningVideo })
            } catch {
              // ignore transient frame errors
            } finally {
              processingRef.current = false
            }
          }

          loopRef.current = window.requestAnimationFrame(() => {
            void loop()
          })
        }

        void loop()
      } catch {
        // no-op: overlay stays usable even if tracking script fails.
      }
    }

    void startTracking()

    return () => {
      cancelled = true
      if (loopRef.current) {
        window.cancelAnimationFrame(loopRef.current)
        loopRef.current = null
      }
      handsRef.current = null
      processingRef.current = false
      setHandDetected(false)
      setLandmarkCount(0)
      clearCanvas()
    }
  }, [isStreaming, videoRef])

  const openOverlay = () => {
    setHidden(false)
    setMinimized(false)
  }

  const closeOverlay = () => {
    setHidden(true)
  }

  if (hidden) {
    return (
      <button type="button" className="overlay-launcher" onClick={openOverlay}>
        <span>Open camera panel</span>
        <small>{isStreaming ? 'Camera active' : 'Camera idle'}</small>
      </button>
    )
  }

  return (
    <>
      <aside className={`camera-overlay ${minimized ? 'minimized' : ''} ${dockSide === 'left' ? 'left' : 'right'}`}>
        <button
          type="button"
          className="overlay-close-btn"
          onClick={closeOverlay}
          aria-label="Close panel"
          title="Close panel"
        >
          x
        </button>
        <button
          type="button"
          className="overlay-min-btn"
          onClick={() => setMinimized((value) => !value)}
          aria-label={minimized ? 'Expand panel' : 'Minimize panel'}
          title={minimized ? 'Expand panel' : 'Minimize panel'}
        >
          {minimized ? '+' : '-'}
        </button>
        <div className="overlay-head">
          <div>
            <p className="eyebrow">Live Camera Panel</p>
            <h3>Tracking and pointer controls</h3>
          </div>
          <div className="overlay-actions">
            <button
              type="button"
              className="ghost-button"
              onClick={() => setDockSide((value) => (value === 'right' ? 'left' : 'right'))}
            >
              Dock {dockSide === 'right' ? 'left' : 'right'}
            </button>
            <button type="button" className="ghost-button" onClick={() => setMinimized((value) => !value)}>
              {minimized ? 'Expand' : 'Minimize'}
            </button>
            <button type="button" className="ghost-button" onClick={closeOverlay}>
              Hide
            </button>
          </div>
        </div>

        {!minimized && (
          <>
            <div className="form-grid compact overlay-form">
              <label className="input-group wide">
                <span>Camera device</span>
                <select className="text-input" value={selectedDeviceId} onChange={(event) => setSelectedDeviceId(event.target.value)}>
                  <option value="">Auto select</option>
                  {devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="camera-frame">
              <video ref={videoRef} autoPlay muted playsInline />
              <canvas ref={canvasRef} className="camera-landmark-canvas" />
              {!isStreaming && (
                <div className="camera-placeholder">
                  <span>Camera idle</span>
                  <p>Start camera to view hand points and pointer tracking.</p>
                </div>
              )}
              {isStreaming && handDetected && pointerMode && (
                <div
                  className="pointer-dot"
                  style={{
                    left: `${pointer.x * 100}%`,
                    top: `${pointer.y * 100}%`,
                  }}
                />
              )}
            </div>

            <div className="overlay-status-grid">
              <div>
                <span>Status</span>
                <strong>{state.toUpperCase()}</strong>
              </div>
              <div>
                <span>Tracking</span>
                <strong>{handDetected ? `HAND DETECTED (${landmarkCount})` : 'NO HAND'}</strong>
              </div>
              <div>
                <span>Backend</span>
                <strong>{backendOnline ? 'ONLINE' : 'OFFLINE'}</strong>
              </div>
            </div>

            <p className="muted-copy overlay-note">
              Browser security does not allow moving your OS cursor directly. This panel gives a virtual pointer and
              landmark tracking for demo and interaction testing.
            </p>

            {error && <p className="inline-error">{error}</p>}

            <div className="overlay-controls">
              <button type="button" className="primary-button" onClick={isStreaming ? stop : start}>
                {isStreaming ? 'Stop camera' : 'Start camera'}
              </button>
              <button type="button" className="secondary-button" onClick={() => void refreshDevices()}>
                Refresh camera list
              </button>
              <button type="button" className="secondary-button" onClick={() => setPointerMode((value) => !value)}>
                {pointerMode ? 'Disable pointer mode' : 'Enable pointer mode'}
              </button>
              <button type="button" className="ghost-button" onClick={closeOverlay}>
                Close panel
              </button>
            </div>
          </>
        )}

        {minimized && (
          <div className="overlay-mini-shell">
            <div className="overlay-mini-status">
              <span>{isStreaming ? 'Camera active' : 'Camera idle'}</span>
              <strong>{handDetected ? 'Hand detected' : 'Show hand to track'}</strong>
            </div>
            {error && <p className="inline-error">{error}</p>}
            <div className="overlay-mini-actions">
              <button type="button" className="primary-button" onClick={isStreaming ? stop : start}>
                {isStreaming ? 'Stop camera' : 'Start camera'}
              </button>
              <button type="button" className="secondary-button" onClick={() => setPointerMode((value) => !value)}>
                {pointerMode ? 'Pointer on' : 'Pointer off'}
              </button>
            </div>
          </div>
        )}
      </aside>

      {isStreaming && handDetected && pointerMode && (
        <div
          className="virtual-pointer"
          style={{
            left: `${pointer.x * 100}%`,
            top: `${pointer.y * 100}%`,
          }}
        />
      )}
    </>
  )
}
