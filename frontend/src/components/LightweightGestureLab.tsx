import { useEffect, useMemo, useRef, useState } from 'react'

type ActionType = 'none' | 'open_url' | 'open_app' | 'hotkey' | 'type_text'
type CameraState = 'idle' | 'loading' | 'streaming' | 'error'

type GestureAction = {
  action_type: ActionType
  value: string
  enabled: boolean
  cooldown_ms: number
  description: string
}

type GestureClassState = {
  label: string
  samples: number
  prototype: number[]
}

type GestureProfile = {
  id: string
  name: string
  labels: string[]
  sequence_length: number
  classes: Record<string, GestureClassState>
  mappings: Record<string, Record<string, GestureAction>>
}

type PredictionResult = {
  label: string | null
  confidence: number
  accepted: boolean
  action: GestureAction | null
}

type LightweightGestureLabProps = {
  backendOnline: boolean
}

type TypingBindings = {
  nextLabel: string
  prevLabel: string
  selectLabel: string
  backspaceLabel: string
  spaceLabel: string
  submitLabel: string
}

declare global {
  interface Window {
    Hands?: any
  }
}

const SCRIPT_URLS = ['https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js']

const KEYBOARD_ROWS = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
]
const KEYBOARD_KEYS = KEYBOARD_ROWS.flat()

const DEFAULT_BINDINGS: TypingBindings = {
  nextLabel: 'fist',
  prevLabel: 'peace',
  selectLabel: 'pinch',
  backspaceLabel: 'open',
  spaceLabel: 'three',
  submitLabel: 'shaka',
}

const SITE_MAP: Record<string, string> = {
  google: 'https://www.google.com',
  youtube: 'https://www.youtube.com',
  github: 'https://github.com',
  gmail: 'https://mail.google.com',
  chatgpt: 'https://chat.openai.com',
  linkedin: 'https://www.linkedin.com',
  wikipedia: 'https://www.wikipedia.org',
  x: 'https://x.com',
  twitter: 'https://x.com',
}

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

const LIVE_POLL_MS = 240
const STABILITY_WINDOW = 3
const STABILITY_REQUIRED = 2
const MIN_EXECUTE_COOLDOWN = 2200
const TRAINING_TARGET_PER_LABEL = 6
const TRAINING_READY_PERCENT = 70
const POINTER_SMOOTHING = 0.72
const ACTION_LOG_LIMIT = 10
const HAND_STALE_MS = 650
const FRAME_STALE_MS = 1200

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

async function ensureMediapipe() {
  for (const url of SCRIPT_URLS) {
    // eslint-disable-next-line no-await-in-loop
    await loadScript(url)
  }
}

function normalizeLabel(value: string) {
  return value.trim().toLowerCase()
}

function queryToUrl(rawQuery: string): string {
  const query = rawQuery.trim()
  if (!query) {
    return ''
  }

  const lower = query.toLowerCase()
  if (lower.startsWith('open ')) {
    const target = lower.slice(5).trim()
    if (!target) {
      return ''
    }
    if (target.startsWith('http://') || target.startsWith('https://')) {
      return target
    }
    if (SITE_MAP[target]) {
      return SITE_MAP[target]
    }
    if (target.includes('.')) {
      return `https://${target}`
    }
  }

  if (lower.startsWith('http://') || lower.startsWith('https://')) {
    return query
  }

  const direct = SITE_MAP[lower]
  if (direct) {
    return direct
  }

  return `https://www.google.com/search?q=${encodeURIComponent(query)}`
}

export function LightweightGestureLab({ backendOnline }: LightweightGestureLabProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const sequenceBufferRef = useRef<number[][][]>([])
  const handsRef = useRef<any>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const detectTimerRef = useRef<number | null>(null)
  const liveLoopRef = useRef<number | null>(null)
  const predictBusyRef = useRef(false)
  const predictionTrailRef = useRef<string[]>([])
  const lastExecutionRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const lastStableNotifyRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const stableGestureLatchRef = useRef('')
  const lastHandSeenAtRef = useRef(0)
  const lastFrameAtRef = useRef(0)
  const selectedKeyRef = useRef(0)
  const typingTextRef = useRef('')
  const lastTypingGestureRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const cameraRunningRef = useRef(false)

  const [profiles, setProfiles] = useState<GestureProfile[]>([])
  const [profileId, setProfileId] = useState('')
  const [profileName, setProfileName] = useState('Laptop Gesture Profile')
  const [profileLabels, setProfileLabels] = useState('pinch,peace,three,open,fist')
  const [sequenceLength, setSequenceLength] = useState(16)

  const [cameraState, setCameraState] = useState<CameraState>('idle')
  const [cameraError, setCameraError] = useState('')
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [handDetected, setHandDetected] = useState(false)
  const [landmarkCount, setLandmarkCount] = useState(0)
  const [pointerMode, setPointerMode] = useState(true)
  const [pointer, setPointer] = useState({ x: 0.5, y: 0.5 })

  const [lastPrediction, setLastPrediction] = useState<PredictionResult | null>(null)
  const [liveMode, setLiveMode] = useState(false)
  const [autoExecute, setAutoExecute] = useState(true)
  const [triggerContext, setTriggerContext] = useState('global')
  const [minConfidence, setMinConfidence] = useState(0.58)
  const [status, setStatus] = useState('Create a lightweight profile, start webcam, and train custom gestures.')
  const [mappingSaved, setMappingSaved] = useState(false)
  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'trained'>('idle')
  const [guidedMode, setGuidedMode] = useState(true)
  const [liveStatus, setLiveStatus] = useState<'idle' | 'watching' | 'detected' | 'executing'>('idle')
  const [activityLog, setActivityLog] = useState<string[]>([])

  const [trainLabel, setTrainLabel] = useState('pinch')
  const [mapLabel, setMapLabel] = useState('pinch')
  const [mapContext, setMapContext] = useState('global')
  const [mapActionType, setMapActionType] = useState<ActionType>('open_url')
  const [mapActionValue, setMapActionValue] = useState('https://www.google.com')
  const [mapCooldown, setMapCooldown] = useState(1500)
  const [mapDescription, setMapDescription] = useState('Open browser quickly')

  const [typingEnabled, setTypingEnabled] = useState(true)
  const [typingText, setTypingText] = useState('')
  const [searchStatus, setSearchStatus] = useState('Gesture typing ready.')
  const [selectedKey, setSelectedKey] = useState(0)
  const [typingBindings, setTypingBindings] = useState<TypingBindings>(DEFAULT_BINDINGS)

  const [collapsed, setCollapsed] = useState({
    profile: false,
    capture: false,
    mapping: false,
    typing: true,
  })

  const labels = useMemo(
    () => profileLabels.split(',').map((item) => normalizeLabel(item)).filter(Boolean),
    [profileLabels],
  )

  const bindingKey = useMemo(
    () => Object.values(typingBindings).join('|'),
    [typingBindings],
  )

  const activeProfile = profiles.find((profile) => profile.id === profileId) || null
  const labelOptions = activeProfile?.labels.length ? activeProfile.labels : labels
  const expectedLabelCount = Math.max(1, labelOptions.length || labels.length)

  const totalTrainedSamples = useMemo(() => {
    if (!activeProfile) {
      return 0
    }
    return Object.values(activeProfile.classes).reduce((total, current) => total + current.samples, 0)
  }, [activeProfile])

  const trainingTargetSamples = useMemo(
    () => expectedLabelCount * TRAINING_TARGET_PER_LABEL,
    [expectedLabelCount],
  )

  const trainingProgressPercent = useMemo(() => {
    if (!profileId || trainingTargetSamples <= 0) {
      return 0
    }
    return Math.min(100, Math.round((totalTrainedSamples / trainingTargetSamples) * 100))
  }, [profileId, totalTrainedSamples, trainingTargetSamples])

  const hasAnyMapping = useMemo(() => {
    if (!activeProfile) {
      return false
    }
    const contexts = Object.values(activeProfile.mappings)
    return contexts.some((contextMap) =>
      Object.values(contextMap).some((action) => action.enabled && action.action_type !== 'none'),
    )
  }, [activeProfile])

  const stepCompletion = useMemo(
    () => ({
      profileCreated: Boolean(profileId),
      trackingReady: cameraState === 'streaming' && handDetected,
      trained: totalTrainedSamples > 0,
      mapped: mappingSaved || hasAnyMapping,
      live: liveMode,
    }),
    [cameraState, handDetected, totalTrainedSamples, mappingSaved, hasAnyMapping, liveMode, profileId],
  )

  const canStartProfile = useMemo(
    () => stepCompletion.profileCreated && stepCompletion.trackingReady && stepCompletion.trained && (!autoExecute || stepCompletion.mapped),
    [stepCompletion, autoExecute],
  )

  const nextAction = useMemo(() => {
    if (!stepCompletion.profileCreated) {
      return 'Step 1: Create profile'
    }
    if (!stepCompletion.trackingReady) {
      return 'Step 2: Start camera and show your hand for tracking'
    }
    if (!stepCompletion.trained) {
      return `Step 3: Capture at least one sample (${trainingProgressPercent}% quality target)`
    }
    if (autoExecute && !stepCompletion.mapped) {
      return 'Step 4: Save at least one gesture mapping'
    }
    if (!stepCompletion.live) {
      return 'Step 5: Start profile live mode'
    }
    return 'Profile live. Test gestures now.'
  }, [stepCompletion, trainingProgressPercent, autoExecute])

  const startProfileHint = useMemo(() => {
    if (!stepCompletion.profileCreated) {
      return 'Create profile first.'
    }
    if (!stepCompletion.trackingReady) {
      return 'Start camera and keep hand visible.'
    }
    if (!stepCompletion.trained) {
      return 'Capture at least one training sample.'
    }
    if (autoExecute && !stepCompletion.mapped) {
      return 'Save mapping or disable Auto execute.'
    }
    return 'Ready to start profile.'
  }, [stepCompletion, autoExecute])

  useEffect(() => {
    setMappingSaved(hasAnyMapping)
  }, [hasAnyMapping, profileId])

  useEffect(() => {
    if (totalTrainedSamples === 0) {
      setRecordingState('idle')
      return
    }
    setRecordingState('trained')
  }, [totalTrainedSamples, profileId])

  useEffect(() => {
    if (!guidedMode) {
      return
    }

    if (!stepCompletion.profileCreated) {
      setCollapsed({
        profile: false,
        capture: true,
        mapping: true,
        typing: true,
      })
      return
    }

    if (!stepCompletion.trained) {
      setCollapsed({
        profile: true,
        capture: false,
        mapping: true,
        typing: true,
      })
      return
    }

    if (!stepCompletion.mapped) {
      setCollapsed({
        profile: true,
        capture: true,
        mapping: false,
        typing: true,
      })
      return
    }

    setCollapsed({
      profile: true,
      capture: true,
      mapping: false,
      typing: false,
    })
  }, [guidedMode, stepCompletion.profileCreated, stepCompletion.trained, stepCompletion.mapped])

  useEffect(() => {
    selectedKeyRef.current = selectedKey
  }, [selectedKey])

  useEffect(() => {
    typingTextRef.current = typingText
  }, [typingText])

  useEffect(() => {
    if (labels.length > 0 && !labels.includes(trainLabel)) {
      setTrainLabel(labels[0])
    }
  }, [labels, trainLabel])

  useEffect(() => {
    if (labels.length > 0 && !labels.includes(mapLabel)) {
      setMapLabel(labels[0])
    }
  }, [labels, mapLabel])

  useEffect(() => {
    if (!backendOnline) {
      return
    }
    void refreshProfiles()
    void refreshVideoDevices()
  }, [backendOnline])

  useEffect(() => {
    if (!labelOptions.length) {
      return
    }
    const fallback = labelOptions[0]
    setTypingBindings((prev) => {
      const nextState: TypingBindings = {
        nextLabel: labelOptions.includes(prev.nextLabel) ? prev.nextLabel : fallback,
        prevLabel: labelOptions.includes(prev.prevLabel) ? prev.prevLabel : fallback,
        selectLabel: labelOptions.includes(prev.selectLabel) ? prev.selectLabel : fallback,
        backspaceLabel: labelOptions.includes(prev.backspaceLabel) ? prev.backspaceLabel : fallback,
        spaceLabel: labelOptions.includes(prev.spaceLabel) ? prev.spaceLabel : fallback,
        submitLabel: labelOptions.includes(prev.submitLabel) ? prev.submitLabel : fallback,
      }
      if (JSON.stringify(nextState) === JSON.stringify(prev)) {
        return prev
      }
      return nextState
    })
  }, [labelOptions])

  useEffect(() => {
    if (!liveMode) {
      if (liveLoopRef.current) {
        window.clearInterval(liveLoopRef.current)
        liveLoopRef.current = null
      }
      predictionTrailRef.current = []
      stableGestureLatchRef.current = ''
      setLiveStatus('idle')
      return
    }

    liveLoopRef.current = window.setInterval(() => {
      void runPrediction(autoExecute)
    }, LIVE_POLL_MS)

    return () => {
      if (liveLoopRef.current) {
        window.clearInterval(liveLoopRef.current)
        liveLoopRef.current = null
      }
    }
  }, [liveMode, autoExecute, profileId, minConfidence, triggerContext, typingEnabled, bindingKey])

  useEffect(() => {
    return () => {
      stopCamera()
      if (liveLoopRef.current) {
        window.clearInterval(liveLoopRef.current)
      }
    }
  }, [])

  async function refreshProfiles() {
    const response = await fetch('/api/light/profiles')
    if (!response.ok) {
      return
    }
    const payload = (await response.json()) as GestureProfile[]
    setProfiles(payload)
    if (!profileId && payload.length > 0) {
      setProfileId(payload[0].id)
      setProfileLabels(payload[0].labels.join(','))
      setSequenceLength(payload[0].sequence_length)
    }
  }

  async function refreshVideoDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const cams = devices.filter((device) => device.kind === 'videoinput')
      setCameraDevices(cams)
      if (!selectedDeviceId && cams[0]) {
        setSelectedDeviceId(cams[0].deviceId)
      }
    } catch {
      // Ignore: some browsers block enumerateDevices before permission.
    }
  }

  function setBinding<K extends keyof TypingBindings>(key: K, value: string) {
    setTypingBindings((prev) => ({ ...prev, [key]: normalizeLabel(value) }))
  }

  function logActivity(message: string) {
    const timestamp = new Date().toLocaleTimeString()
    setActivityLog((prev) => [`${timestamp} - ${message}`, ...prev].slice(0, ACTION_LOG_LIMIT))
  }

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
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) {
      return
    }

    const bounds = video.getBoundingClientRect()
    if (bounds.width < 2 || bounds.height < 2) {
      return
    }

    const ratio = window.devicePixelRatio || 1
    const targetWidth = Math.floor(bounds.width * ratio)
    const targetHeight = Math.floor(bounds.height * ratio)
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
      context.moveTo((1 - startPoint.x) * bounds.width, startPoint.y * bounds.height)
      context.lineTo((1 - endPoint.x) * bounds.width, endPoint.y * bounds.height)
      context.stroke()
    })

    context.fillStyle = 'rgba(127, 209, 255, 0.98)'
    points.forEach((point, index) => {
      const radius = index === 8 ? 6 : 4
      context.beginPath()
      context.arc((1 - point.x) * bounds.width, point.y * bounds.height, radius, 0, Math.PI * 2)
      context.fill()
    })
  }

  function toggleSection(section: keyof typeof collapsed) {
    setCollapsed((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  async function createProfile() {
    if (!backendOnline) {
      setStatus('Backend is offline. Start backend first.')
      return
    }
    if (labels.length < 2) {
      setStatus('Add at least 2 gesture labels.')
      return
    }

    const response = await fetch('/api/light/profiles', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: profileName,
        labels,
        sequence_length: sequenceLength,
      }),
    })

    if (!response.ok) {
      setStatus('Could not create profile.')
      return
    }

    const profile = (await response.json()) as GestureProfile
    setProfileId(profile.id)
    setProfiles((prev) => [...prev, profile])
    setMappingSaved(false)
    setRecordingState('idle')
    setStatus(`Profile ${profile.id} created.`)
    logActivity(`Profile created (${profile.id}).`)
  }

  function handleLiveModeToggle(nextState: boolean) {
    if (!nextState) {
      setLiveMode(false)
      setLiveStatus('idle')
      setStatus('Live mode stopped.')
      logActivity('Profile stopped.')
      return
    }

    if (!stepCompletion.profileCreated) {
      setStatus('Create profile first before starting live mode.')
      return
    }
    if (!stepCompletion.trackingReady) {
      setStatus('Start webcam and keep hand in frame before starting live mode.')
      return
    }
    if (!stepCompletion.trained) {
      setStatus('Capture more training samples before starting live mode.')
      return
    }
    if (autoExecute && !stepCompletion.mapped) {
      setStatus('Save at least one mapping, or disable Auto execute and retry Start profile.')
      return
    }

    setLiveMode(true)
    setLiveStatus('watching')
    if (trainingProgressPercent < TRAINING_READY_PERCENT) {
      setStatus(`Profile live mode started in low-data mode (${trainingProgressPercent}% quality).`)
      logActivity(`Profile started in low-data mode (${trainingProgressPercent}%).`)
      return
    }
    setStatus('Profile live mode started.')
    logActivity('Profile started.')
  }

  function stopCamera() {
    cameraRunningRef.current = false
    if (detectTimerRef.current) {
      window.clearTimeout(detectTimerRef.current)
      detectTimerRef.current = null
    }

    try {
      streamRef.current?.getTracks().forEach((track) => track.stop())
    } catch {
      // no-op
    }

    streamRef.current = null
    handsRef.current = null
    sequenceBufferRef.current = []
    predictionTrailRef.current = []
    stableGestureLatchRef.current = ''
    lastHandSeenAtRef.current = 0
    lastFrameAtRef.current = 0
    setLiveStatus('idle')
    setHandDetected(false)
    setLandmarkCount(0)
    clearCanvas()

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    if (cameraState !== 'idle') {
      setCameraState('idle')
    }
  }

  async function startCamera() {
    setCameraError('')
    setCameraState('loading')
    stopCamera()

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API is not available in this browser.')
      }

      await ensureMediapipe()
      const video = videoRef.current
      if (!video || !window.Hands) {
        throw new Error('MediaPipe Hands is unavailable.')
      }

      const attempts: MediaStreamConstraints[] = []
      if (selectedDeviceId) {
        attempts.push({
          video: {
            deviceId: { exact: selectedDeviceId },
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        })
      }
      attempts.push({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      })
      attempts.push({ video: true, audio: false })

      let stream: MediaStream | null = null
      let lastErr: unknown = null
      for (const constraints of attempts) {
        try {
          // eslint-disable-next-line no-await-in-loop
          stream = await navigator.mediaDevices.getUserMedia(constraints)
          break
        } catch (err) {
          lastErr = err
        }
      }

      if (!stream) {
        throw lastErr ?? new Error('Unable to access any webcam device.')
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
        lastFrameAtRef.current = Date.now()
        const detected = results?.multiHandLandmarks?.[0] as Array<{ x: number; y: number; z?: number }> | undefined
        if (!detected) {
          sequenceBufferRef.current = []
          predictionTrailRef.current = []
          stableGestureLatchRef.current = ''
          lastHandSeenAtRef.current = 0
          setHandDetected(false)
          setLandmarkCount(0)
          drawLandmarks(null)
          return
        }

        lastHandSeenAtRef.current = Date.now()
        setHandDetected(true)
        setLandmarkCount(detected.length)
        if (detected[8]) {
          const targetX = Math.max(0, Math.min(1, 1 - detected[8].x))
          const targetY = Math.max(0, Math.min(1, detected[8].y))
          setPointer((prev) => ({
            x: prev.x + (targetX - prev.x) * POINTER_SMOOTHING,
            y: prev.y + (targetY - prev.y) * POINTER_SMOOTHING,
          }))
        }
        drawLandmarks(detected)

        const frame = detected.slice(0, 21).map((point: any) => [point.x, point.y, point.z ?? 0])
        sequenceBufferRef.current.push(frame)
        if (sequenceBufferRef.current.length > 90) {
          sequenceBufferRef.current.shift()
        }
      })

      streamRef.current = stream
      video.srcObject = stream
      await video.play().catch(() => undefined)

      handsRef.current = hands
      cameraRunningRef.current = true

      const detectLoop = async () => {
        if (!cameraRunningRef.current) {
          return
        }
        const runningVideo = videoRef.current
        const runningHands = handsRef.current
        if (runningVideo && runningHands && runningVideo.readyState >= 2) {
          try {
            await runningHands.send({ image: runningVideo })
          } catch {
            // ignore frame-level errors
          }
        }
        detectTimerRef.current = window.setTimeout(() => {
          void detectLoop()
        }, 70)
      }

      void detectLoop()
      setCameraState('streaming')
      setLiveStatus('watching')
      setStatus('Webcam streaming. Capture gestures for training.')
      logActivity('Camera started. Tracking active.')
      await refreshVideoDevices()
    } catch (cameraErr) {
      const message = cameraErr instanceof Error ? cameraErr.message : 'Failed to start camera.'
      const userMessage = /requested device not found|notfounderror|device/i.test(message)
        ? 'Requested camera device not found. Select another camera or unplug/plug webcam, then retry.'
        : message
      setCameraError(userMessage)
      setCameraState('error')
      setLiveStatus('idle')
      setStatus('Camera initialization failed.')
      setHandDetected(false)
      setLandmarkCount(0)
      clearCanvas()
      logActivity(`Camera error: ${userMessage}`)
    }
  }

  function getWindowSequence() {
    const buffer = sequenceBufferRef.current
    if (buffer.length < 8) {
      return null
    }
    return buffer.slice(Math.max(0, buffer.length - sequenceLength))
  }

  async function captureTrainSample() {
    if (!profileId) {
      setStatus('Create/select a profile first.')
      return
    }

    const sequence = getWindowSequence()
    if (!sequence) {
      setStatus('Not enough webcam frames yet. Keep your hand visible for a moment.')
      return
    }

    setRecordingState('recording')
    setStatus('Recording sample...')
    logActivity(`Recording sample for gesture: ${trainLabel}`)

    const response = await fetch('/api/light/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: profileId,
        label: trainLabel,
        sequence,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Training failed' }))
      setRecordingState('idle')
      setStatus(err.detail ?? 'Training failed.')
      return
    }

    const payload = await response.json()
    setRecordingState('trained')
    const projectedProgress = Math.min(100, Math.round(((totalTrainedSamples + 1) / Math.max(1, trainingTargetSamples)) * 100))
    setStatus(`Captured ${trainLabel}. Samples for label: ${payload.samples}. Training progress: ${projectedProgress}%`)
    logActivity(`Trained ${trainLabel}. Progress ${projectedProgress}%`)
    await refreshProfiles()
  }

  function appendTyping(value: string) {
    setTypingText((prev) => prev + value)
  }

  function backspaceTyping() {
    setTypingText((prev) => prev.slice(0, -1))
  }

  function executeSearch(inputText?: string) {
    const query = (inputText ?? typingTextRef.current).trim()
    if (!query) {
      setSearchStatus('Type text first, then run search/open command.')
      return
    }

    const url = queryToUrl(query)
    if (!url) {
      setSearchStatus('Could not resolve command.')
      return
    }

    const opened = window.open(url, '_blank', 'noopener,noreferrer')
    if (!opened) {
      window.location.href = url
    }
    setSearchStatus(`Opened: ${url}`)
  }

  function applyTypingGesture(label: string) {
    if (!typingEnabled) {
      return
    }

    const normalized = normalizeLabel(label)
    const now = Date.now()
    if (lastTypingGestureRef.current.label === normalized && now - lastTypingGestureRef.current.ts < 700) {
      return
    }
    lastTypingGestureRef.current = { label: normalized, ts: now }

    if (normalized === typingBindings.nextLabel) {
      setSelectedKey((prev) => (prev + 1) % KEYBOARD_KEYS.length)
      return
    }
    if (normalized === typingBindings.prevLabel) {
      setSelectedKey((prev) => (prev - 1 + KEYBOARD_KEYS.length) % KEYBOARD_KEYS.length)
      return
    }
    if (normalized === typingBindings.selectLabel) {
      const key = KEYBOARD_KEYS[selectedKeyRef.current]
      if (key) {
        appendTyping(key)
      }
      return
    }
    if (normalized === typingBindings.backspaceLabel) {
      backspaceTyping()
      return
    }
    if (normalized === typingBindings.spaceLabel) {
      appendTyping(' ')
      return
    }
    if (normalized === typingBindings.submitLabel) {
      executeSearch()
    }
  }

  async function runPrediction(executeMapped: boolean) {
    if (!profileId || predictBusyRef.current) {
      return
    }

    const nowTime = Date.now()
    const handFresh = lastHandSeenAtRef.current > 0 && nowTime - lastHandSeenAtRef.current <= HAND_STALE_MS
    const frameFresh = lastFrameAtRef.current > 0 && nowTime - lastFrameAtRef.current <= FRAME_STALE_MS

    if (!handDetected || !handFresh || !frameFresh) {
      predictionTrailRef.current = []
      stableGestureLatchRef.current = ''
      if (liveMode) {
        setLiveStatus('watching')
      }
      return
    }

    const sequence = getWindowSequence()
    if (!sequence) {
      predictionTrailRef.current = []
      stableGestureLatchRef.current = ''
      if (liveMode) {
        setLiveStatus('watching')
      }
      return
    }

    predictBusyRef.current = true
    try {
      const response = await fetch('/api/light/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          sequence,
          context: triggerContext,
          min_confidence: minConfidence,
        }),
      })

      if (!response.ok) {
        return
      }

      const prediction = (await response.json()) as PredictionResult
      setLastPrediction(prediction)

      if (!prediction.accepted || !prediction.label) {
        predictionTrailRef.current = []
        if (liveMode) {
          setLiveStatus('watching')
        }
        return
      }

      const normalized = normalizeLabel(prediction.label)
      predictionTrailRef.current = [...predictionTrailRef.current.slice(-(STABILITY_WINDOW - 1)), normalized]
      const stableVotes = predictionTrailRef.current.filter((item) => item === normalized).length
      const stableAccepted = stableVotes >= STABILITY_REQUIRED

      if (!stableAccepted) {
        if (liveMode) {
          setLiveStatus('detected')
        }
        return
      }

      const isNewStableGesture = stableGestureLatchRef.current !== normalized
      if (!isNewStableGesture) {
        return
      }
      stableGestureLatchRef.current = normalized

      if (liveMode) {
        setLiveStatus('detected')
      }

      const now = Date.now()
      if (
        lastStableNotifyRef.current.label !== normalized
        || now - lastStableNotifyRef.current.ts > 1400
      ) {
        lastStableNotifyRef.current = {
          label: normalized,
          ts: now,
        }
        setStatus(`Detected gesture: ${normalized} (${(prediction.confidence * 100).toFixed(1)}%)`)
        logActivity(`Detected ${normalized} (${(prediction.confidence * 100).toFixed(1)}%)`)
      }

      applyTypingGesture(normalized)

      if (!executeMapped) {
        return
      }

      if (!prediction.action?.enabled) {
        setStatus(`Detected ${normalized}, but no enabled action is mapped.`)
        logActivity(`Detected ${normalized}, but no mapping is enabled.`)
        return
      }

      const effectiveCooldown = Math.max(prediction.action.cooldown_ms || 0, MIN_EXECUTE_COOLDOWN)
      if (
        lastExecutionRef.current.label === normalized
        && now - lastExecutionRef.current.ts < effectiveCooldown
      ) {
        setStatus(`Detected ${normalized}, waiting cooldown...`)
        return
      }

      if (liveMode) {
        setLiveStatus('executing')
      }

      const executeResponse = await fetch('/api/light/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          context: triggerContext,
          label: normalized,
        }),
      })

      if (executeResponse.ok) {
        const result = await executeResponse.json()
        lastExecutionRef.current = {
          label: normalized,
          ts: now,
        }
        setStatus(result.detail)
        logActivity(`Action: ${result.detail}`)
        if (liveMode) {
          setLiveStatus('watching')
        }
      } else {
        setStatus(`Action failed for ${normalized}`)
        if (liveMode) {
          setLiveStatus('watching')
        }
      }
    } finally {
      predictBusyRef.current = false
    }
  }

  async function saveMapping() {
    if (!profileId) {
      setStatus('Create/select a profile first.')
      return
    }

    const response = await fetch('/api/light/mappings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: profileId,
        context: mapContext,
        label: mapLabel,
        action_type: mapActionType,
        value: mapActionValue,
        enabled: true,
        cooldown_ms: mapCooldown,
        description: mapDescription,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to save mapping' }))
      setStatus(err.detail ?? 'Failed to save mapping')
      return
    }

    await refreshProfiles()
    setMappingSaved(true)
    setStatus(`Mapping saved: ${mapLabel} -> ${mapActionType}`)
    logActivity(`Mapping saved for ${mapLabel} (${mapActionType}).`)
  }

  return (
    <section className="gesture-lab-shell" id="lightweight-lab">
      <div className="section-heading">
        <p className="eyebrow">Lightweight dynamic trainer</p>
        <h3>Train custom gestures from webcam and control browser/app actions</h3>
      </div>
      <div className="workflow-shell">
        <div className="workflow-head">
          <div>
            <p className="eyebrow">Guided workflow</p>
            <h4>Start gesture, then tracking, then recording, then recognition complete, then map action, then start profile</h4>
          </div>
          <div className="workflow-controls">
            <span className={`engine-pill ${liveStatus}`}>Engine: {liveStatus.toUpperCase()}</span>
            <label className="pointer-toggle">
              <input type="checkbox" checked={guidedMode} onChange={(e) => setGuidedMode(e.target.checked)} />
              Guided mode
            </label>
          </div>
        </div>
        <div className="workflow-grid" role="status" aria-live="polite">
          <div className={`workflow-item ${stepCompletion.profileCreated ? 'done' : 'active'}`}>
            <span>1</span>
            <p>Start gesture profile</p>
          </div>
          <div className={`workflow-item ${stepCompletion.trackingReady ? 'done' : stepCompletion.profileCreated ? 'active' : 'pending'}`}>
            <span>2</span>
            <p>Tracking ready</p>
          </div>
          <div className={`workflow-item ${recordingState === 'trained' ? 'done' : recordingState === 'recording' ? 'active' : 'pending'}`}>
            <span>3</span>
            <p>{recordingState === 'recording' ? 'Recording now' : 'Recording'}</p>
          </div>
          <div className={`workflow-item ${stepCompletion.trained ? 'done' : 'pending'}`}>
            <span>4</span>
            <p>Recognition complete</p>
          </div>
          <div className={`workflow-item ${stepCompletion.mapped ? 'done' : 'pending'}`}>
            <span>5</span>
            <p>Map action</p>
          </div>
          <div className={`workflow-item ${stepCompletion.live ? 'done' : 'pending'}`}>
            <span>6</span>
            <p>Start profile</p>
          </div>
        </div>
        <div className="train-progress-shell" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={trainingProgressPercent}>
          <div className="train-progress-head">
            <span>Training progress</span>
            <strong>{trainingProgressPercent}%</strong>
          </div>
          <div className="train-progress-track">
            <div className="train-progress-fill" style={{ width: `${trainingProgressPercent}%` }} />
          </div>
          <p className="train-progress-caption">
            {totalTrainedSamples}/{trainingTargetSamples} samples captured. Minimum 1 sample enables live mode; {TRAINING_READY_PERCENT}% is recommended quality.
          </p>
        </div>
        <p className="next-action-line">Next: {nextAction}</p>
      </div>
      <p className="lab-helper">Order: Step 1 create profile, then step 2 start camera and capture, then step 3 map action, then step 4 enable live mode.</p>
      <p className="lab-note">
        Dev mode uses two ports: UI on <code>5173</code> and API on <code>8000</code>. For single URL mode,
        build frontend and open <code>http://127.0.0.1:8000/studio</code>.
      </p>
      <p className="scope-note">
        Gesture pointer and controls work inside this app tab. Browser security blocks direct full-system mouse/scroll
        control on any website unless you build a browser extension or desktop background service.
      </p>

      <div className="gesture-lab-grid">
        <article className="glass-card gesture-lab-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Profile</p>
              <h4>Create a lightweight profile</h4>
            </div>
            <button type="button" className="mini-toggle" onClick={() => toggleSection('profile')}>
              {guidedMode ? (stepCompletion.profileCreated ? 'Done' : 'Now') : collapsed.profile ? 'Expand' : 'Collapse'}
            </button>
          </div>

          {!collapsed.profile && (
            <>
              <div className="form-grid compact">
                <label className="input-group wide">
                  <span>Profile name</span>
                  <input
                    className="text-input"
                    aria-label="Profile name"
                    value={profileName}
                    onChange={(e) => setProfileName(e.target.value)}
                  />
                </label>

                <label className="input-group wide">
                  <span>Gesture labels</span>
                  <input
                    className="text-input"
                    aria-label="Gesture labels"
                    value={profileLabels}
                    onChange={(e) => setProfileLabels(e.target.value)}
                    placeholder="pinch,peace,three,open,fist"
                  />
                </label>

                <label className="input-group">
                  <span>Sequence length</span>
                  <input
                    className="text-input"
                    type="number"
                    min={8}
                    max={64}
                    value={sequenceLength}
                    onChange={(e) => setSequenceLength(Number(e.target.value))}
                  />
                </label>

                <label className="input-group">
                  <span>Active profile</span>
                  <select className="text-input" value={profileId} onChange={(e) => setProfileId(e.target.value)}>
                    <option value="">Select profile</option>
                    {profiles.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.name} ({profile.id})
                      </option>
                    ))}
                  </select>
                </label>

                <div className="action-row wide">
                  <button type="button" className="primary-button" onClick={createProfile}>
                    Create profile
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void refreshProfiles()}>
                    Refresh profiles
                  </button>
                </div>
              </div>

              <div className="profile-summary">
                <span>Trained classes</span>
                <strong>{activeProfile ? Object.keys(activeProfile.classes).length : 0}</strong>
              </div>
            </>
          )}
        </article>

        <article className="glass-card gesture-lab-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Webcam capture</p>
              <h4>Capture and train gesture samples</h4>
            </div>
            <button type="button" className="mini-toggle" onClick={() => toggleSection('capture')}>
              {guidedMode ? (stepCompletion.trained ? 'Done' : 'Now') : collapsed.capture ? 'Expand' : 'Collapse'}
            </button>
          </div>

          {!collapsed.capture && (
            <>
              <div className="form-grid compact">
                <label className="input-group wide">
                  <span>Camera device</span>
                  <select className="text-input" value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)}>
                    <option value="">Auto select</option>
                    {cameraDevices.map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="webcam-box">
                <video ref={videoRef} autoPlay muted playsInline />
                <canvas ref={canvasRef} className="webcam-overlay-canvas" />
                {cameraState === 'streaming' && handDetected && pointerMode && (
                  <div
                    className="hand-cursor"
                    style={{
                      left: `${pointer.x * 100}%`,
                      top: `${pointer.y * 100}%`,
                    }}
                  />
                )}
                {cameraState !== 'streaming' && <div className="webcam-mask">Camera {cameraState.toUpperCase()}</div>}
              </div>

              <div className="tracking-row">
                <span className={`tracking-chip ${handDetected ? 'ok' : 'warn'}`}>
                  {handDetected ? `Tracking active (${landmarkCount} points)` : 'No hand detected'}
                </span>
                <label className="pointer-toggle">
                  <input type="checkbox" checked={pointerMode} onChange={(e) => setPointerMode(e.target.checked)} />
                  Show pointer
                </label>
              </div>
              <p className="mini-hint-line">
                {recordingState === 'recording'
                  ? 'Recording sample...'
                  : recordingState === 'trained'
                    ? `Trained samples: ${totalTrainedSamples}`
                    : 'Press Capture + train to start recording samples.'}
              </p>

              <div className="action-row">
                <button
                  type="button"
                  className="primary-button"
                  onClick={cameraState === 'streaming' ? stopCamera : () => void startCamera()}
                >
                  {cameraState === 'streaming' ? 'Stop webcam' : 'Start webcam'}
                </button>
                <button type="button" className="secondary-button" onClick={() => void refreshVideoDevices()}>
                  Refresh camera list
                </button>
              </div>

              {cameraError && <p className="inline-error">{cameraError}</p>}

              <div className="form-grid compact">
                <label className="input-group">
                  <span>Train label</span>
                  <select className="text-input" value={trainLabel} onChange={(e) => setTrainLabel(e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Confidence threshold</span>
                  <input
                    className="text-input"
                    type="number"
                    min={0.2}
                    max={0.95}
                    step={0.01}
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                  />
                </label>

                <div className="action-row wide">
                  <button type="button" className="secondary-button" onClick={() => void captureTrainSample()}>
                    Capture + train
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void runPrediction(false)}>
                    Predict once
                  </button>
                </div>
              </div>

              <div className="prediction-box">
                <span>Last prediction</span>
                <strong>
                  {lastPrediction?.label ? `${lastPrediction.label} (${(lastPrediction.confidence * 100).toFixed(1)}%)` : '—'}
                </strong>
                <p>{lastPrediction?.accepted ? 'Prediction accepted' : 'Waiting or below threshold'}</p>
              </div>
            </>
          )}
        </article>

        <article className="glass-card gesture-lab-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Action mapping</p>
              <h4>Map gestures to browser or app actions</h4>
            </div>
            <button type="button" className="mini-toggle" onClick={() => toggleSection('mapping')}>
              {guidedMode ? (stepCompletion.mapped ? 'Done' : 'Now') : collapsed.mapping ? 'Expand' : 'Collapse'}
            </button>
          </div>

          {!collapsed.mapping && (
            <>
              <div className="form-grid compact">
                <label className="input-group">
                  <span>Context</span>
                  <select className="text-input" value={mapContext} onChange={(e) => setMapContext(e.target.value)}>
                    <option value="global">global</option>
                    <option value="browser">browser</option>
                    <option value="presentation">presentation</option>
                  </select>
                </label>

                <label className="input-group">
                  <span>Gesture label</span>
                  <select className="text-input" value={mapLabel} onChange={(e) => setMapLabel(e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Action type</span>
                  <select className="text-input" value={mapActionType} onChange={(e) => setMapActionType(e.target.value as ActionType)}>
                    <option value="none">none</option>
                    <option value="open_url">open_url</option>
                    <option value="open_app">open_app</option>
                    <option value="hotkey">hotkey</option>
                    <option value="type_text">type_text</option>
                  </select>
                </label>

                <label className="input-group">
                  <span>Value</span>
                  <input className="text-input" value={mapActionValue} onChange={(e) => setMapActionValue(e.target.value)} />
                </label>

                <label className="input-group">
                  <span>Cooldown ms</span>
                  <input
                    className="text-input"
                    type="number"
                    min={100}
                    max={60000}
                    value={mapCooldown}
                    onChange={(e) => setMapCooldown(Number(e.target.value))}
                  />
                </label>

                <label className="input-group">
                  <span>Description</span>
                  <input className="text-input" value={mapDescription} onChange={(e) => setMapDescription(e.target.value)} />
                </label>

                <div className="action-row wide">
                  <button type="button" className="secondary-button" onClick={() => void saveMapping()}>
                    Save mapping
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void runPrediction(true)}>
                    Predict + execute once
                  </button>
                </div>
              </div>

              <div className="live-controls">
                <label>
                  <input type="checkbox" checked={autoExecute} onChange={(e) => setAutoExecute(e.target.checked)} />
                  Auto execute mapped action
                </label>
                <label>
                  <input type="checkbox" checked={typingEnabled} onChange={(e) => setTypingEnabled(e.target.checked)} />
                  Enable gesture typing controls
                </label>
              </div>

              <div className="start-profile-row">
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => handleLiveModeToggle(true)}
                  disabled={!canStartProfile || liveMode}
                >
                  {liveMode ? 'Profile running' : 'Start profile'}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => handleLiveModeToggle(false)}
                  disabled={!liveMode}
                >
                  Stop profile
                </button>
                <span className={`profile-state-pill ${liveMode ? 'live' : 'stopped'}`}>
                  {liveMode ? 'LIVE' : 'STOPPED'}
                </span>
              </div>
              <p className="start-hint-line">{startProfileHint}</p>

              <label className="input-group">
                <span>Live context</span>
                <select className="text-input" value={triggerContext} onChange={(e) => setTriggerContext(e.target.value)}>
                  <option value="global">global</option>
                  <option value="browser">browser</option>
                  <option value="presentation">presentation</option>
                </select>
              </label>
            </>
          )}
        </article>

        <article className="glass-card gesture-lab-card typing-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Gesture typing + search</p>
              <h4>Type with gestures and run open/search commands</h4>
            </div>
            <button type="button" className="mini-toggle" onClick={() => toggleSection('typing')}>
              {collapsed.typing ? 'Expand' : 'Collapse'}
            </button>
          </div>

          {!collapsed.typing && (
            <>
              <div className="binding-grid">
                <label className="input-group">
                  <span>Next key gesture</span>
                  <select className="text-input" value={typingBindings.nextLabel} onChange={(e) => setBinding('nextLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Previous key gesture</span>
                  <select className="text-input" value={typingBindings.prevLabel} onChange={(e) => setBinding('prevLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Select key gesture</span>
                  <select className="text-input" value={typingBindings.selectLabel} onChange={(e) => setBinding('selectLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Backspace gesture</span>
                  <select className="text-input" value={typingBindings.backspaceLabel} onChange={(e) => setBinding('backspaceLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Space gesture</span>
                  <select className="text-input" value={typingBindings.spaceLabel} onChange={(e) => setBinding('spaceLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Submit gesture</span>
                  <select className="text-input" value={typingBindings.submitLabel} onChange={(e) => setBinding('submitLabel', e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <label className="input-group typing-input-group">
                <span>Type / search field</span>
                <textarea
                  className="typing-field"
                  value={typingText}
                  onChange={(e) => setTypingText(e.target.value)}
                  rows={3}
                  placeholder="Type manually, or use gestures and on-screen keyboard"
                />
              </label>

              <div className="quick-actions">
                <button type="button" className="secondary-button" onClick={() => executeSearch()}>
                  Run search or open command
                </button>
                <button type="button" className="secondary-button" onClick={() => setTypingText('')}>
                  Clear text
                </button>
                <button type="button" className="secondary-button" onClick={backspaceTyping}>
                  Backspace
                </button>
                <button type="button" className="secondary-button" onClick={() => appendTyping(' ')}>
                  Add space
                </button>
              </div>

              <p className="status-line">{searchStatus}</p>

              <div className="keyboard-grid" role="group" aria-label="On-screen keyboard">
                {KEYBOARD_KEYS.map((key, index) => (
                  <button
                    key={`${key}-${index}`}
                    type="button"
                    className={`key-btn ${selectedKey === index ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedKey(index)
                      appendTyping(key)
                    }}
                  >
                    {key}
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="activity-shell" role="log" aria-live="polite">
            <div className="activity-head">
              <span>Live activity</span>
              <strong>{liveStatus.toUpperCase()}</strong>
            </div>
            {activityLog.length === 0 ? (
              <p className="activity-empty">No activity yet. Start profile and show a trained gesture.</p>
            ) : (
              <ul className="activity-list">
                {activityLog.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            )}
          </div>

          <p className="status-line">{status}</p>
        </article>
      </div>
    </section>
  )
}
