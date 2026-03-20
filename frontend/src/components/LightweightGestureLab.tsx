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

type TrainClipResult = {
  profile_id: string
  label: string
  samples_added: number
  samples: number
  total_classes: number
  clip_frames: number
}

type DeepModelStatus = {
  profile_id: string
  available: boolean
  trained_at?: string | null
  backbone?: string | null
  temporal_head?: string | null
  labels: string[]
  samples: number
  detail: string
}

type ModelEngine = 'lightweight' | 'deep'

type ActionPreset = {
  id: string
  name: string
  action_type: ActionType
  value: string
  description: string
}

type MotionTrainingPreset = {
  id: 'free' | 'swipe_down' | 'swipe_up' | 'swipe_left' | 'swipe_right'
  name: string
  instruction: string
  extraDurationMs: number
  sampleCount: number
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

const ACTION_PRESETS: ActionPreset[] = [
  {
    id: 'close-window',
    name: 'Close active window (Alt+F4)',
    action_type: 'hotkey',
    value: 'alt+f4',
    description: 'Close the current window.',
  },
  {
    id: 'minimize-window',
    name: 'Minimize active window (Win+Down)',
    action_type: 'hotkey',
    value: 'win+down',
    description: 'Minimize the active window.',
  },
  {
    id: 'switch-app',
    name: 'Switch app (Alt+Tab)',
    action_type: 'hotkey',
    value: 'alt+tab',
    description: 'Switch to next application.',
  },
  {
    id: 'next-tab',
    name: 'Next browser tab (Ctrl+Tab)',
    action_type: 'hotkey',
    value: 'ctrl+tab',
    description: 'Move to next browser tab.',
  },
  {
    id: 'prev-tab',
    name: 'Previous browser tab (Ctrl+Shift+Tab)',
    action_type: 'hotkey',
    value: 'ctrl+shift+tab',
    description: 'Move to previous browser tab.',
  },
  {
    id: 'gmail-compose',
    name: 'Open Gmail compose',
    action_type: 'open_url',
    value: 'https://mail.google.com/mail/u/0/#inbox?compose=new',
    description: 'Open Gmail compose window.',
  },
  {
    id: 'open-calculator',
    name: 'Open Calculator',
    action_type: 'open_app',
    value: 'calc.exe',
    description: 'Launch Calculator.',
  },
  {
    id: 'open-notepad',
    name: 'Open Notepad',
    action_type: 'open_app',
    value: 'notepad.exe',
    description: 'Launch Notepad.',
  },
]

const MOTION_TRAINING_PRESETS: MotionTrainingPreset[] = [
  {
    id: 'free',
    name: 'Free motion clip',
    instruction: 'Perform your gesture naturally for the full clip.',
    extraDurationMs: 0,
    sampleCount: 6,
  },
  {
    id: 'swipe_down',
    name: 'Swipe down (top -> bottom)',
    instruction: 'Start near top, move your hand down in one clean motion.',
    extraDurationMs: 700,
    sampleCount: 8,
  },
  {
    id: 'swipe_up',
    name: 'Swipe up (bottom -> top)',
    instruction: 'Start low, move your hand up smoothly.',
    extraDurationMs: 700,
    sampleCount: 8,
  },
  {
    id: 'swipe_left',
    name: 'Swipe left (right -> left)',
    instruction: 'Start from right side and move left with steady speed.',
    extraDurationMs: 700,
    sampleCount: 8,
  },
  {
    id: 'swipe_right',
    name: 'Swipe right (left -> right)',
    instruction: 'Start from left side and move right with steady speed.',
    extraDurationMs: 700,
    sampleCount: 8,
  },
]

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

const LIVE_POLL_MS = 240
const DETECT_LOOP_MS = 40
const PROFILE_SYNC_MS = 3500
const STABILITY_WINDOW = 3
const STABILITY_REQUIRED = 2
const NON_PINCH_STABILITY_EXTRA = 1
const MIN_EXECUTE_COOLDOWN = 2200
const EXECUTE_HOLD_MS = 320
const EXECUTE_CONFIDENCE_MARGIN_NON_PINCH = 0.12
const EXECUTE_CONFIDENCE_MARGIN_PINCH = 0.04
const PINCH_INTENT_CONFIDENCE_ASSIST = 0.08
const TRAINING_TARGET_PER_LABEL = 6
const TRAINING_READY_PERCENT = 70
const POINTER_FAST_ALPHA = 0.58
const POINTER_SLOW_ALPHA = 0.24
const POINTER_DEADZONE = 0.009
const ACTION_LOG_LIMIT = 10
const HAND_STALE_MS = 650
const FRAME_STALE_MS = 1200
const LIVE_TRAIN_DURATION_MS = 2200
const ITERATIVE_CAPTURE_DEFAULT = 2
const ITERATIVE_CAPTURE_MAX = 6
const ITERATIVE_CAPTURE_GAP_MIN_MS = 350
const ITERATIVE_CAPTURE_GAP_DEFAULT_MS = 1200
const POINTER_LOCK_MS_ON_PINCH = 180
const POINTER_LOCK_ALPHA = 0.08
const PINCH_RATIO_ENTER = 0.37
const PINCH_RATIO_CONFIRM = 0.335
const PINCH_RATIO_RELEASE = 0.5
const PINCH_RATIO_TRANSITION = 0.44
const PINCH_RATIO_VELOCITY = -0.006
const PINCH_STABLE_FRAMES = 2
const ZERO_ZONE_LABELS = new Set(['fist', 'closed', 'closed_hand', 'closed_paw', 'close_paw'])

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

function normalizeContext(value: string) {
  const cleaned = value.trim().toLowerCase()
  if (!cleaned) {
    return 'global'
  }
  if (cleaned.startsWith('site:')) {
    let host = cleaned.slice(5).trim()
    host = host.replace(/^https?:\/\//, '').split('/')[0] ?? ''
    host = host.replace(/^www\./, '')
    return host ? `site:${host}` : 'browser'
  }
  return cleaned
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip" title={text} aria-label={text}>
      i
    </span>
  )
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

function isZeroZoneLabel(label: string) {
  const normalized = normalizeLabel(label)
  return ZERO_ZONE_LABELS.has(normalized)
}

function detectClosedPawPose(points: Array<{ x: number; y: number; z?: number }>) {
  if (!points || points.length < 21) {
    return false
  }

  const distance3 = (a: { x: number; y: number; z?: number }, b: { x: number; y: number; z?: number }) => {
    const dx = a.x - b.x
    const dy = a.y - b.y
    const dz = (a.z ?? 0) - (b.z ?? 0)
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }

  const wrist = points[0]
  const fingerPairs: Array<[number, number]> = [
    [8, 6],
    [12, 10],
    [16, 14],
    [20, 18],
  ]

  let foldedCount = 0
  for (const [tipIndex, pipIndex] of fingerPairs) {
    const tipDistance = distance3(points[tipIndex], wrist)
    const pipDistance = distance3(points[pipIndex], wrist)
    if (tipDistance <= pipDistance * 1.03) {
      foldedCount += 1
    }
  }

  const palmWidth = distance3(points[5], points[17])
  const thumbSpan = distance3(points[4], points[5])
  const thumbFolded = thumbSpan <= palmWidth * 0.62

  return foldedCount >= 3 && thumbFolded
}

export function LightweightGestureLab({ backendOnline }: LightweightGestureLabProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const sequenceBufferRef = useRef<number[][][]>([])
  const handsRef = useRef<any>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const detectLoopRef = useRef<number | null>(null)
  const detectProcessingRef = useRef(false)
  const detectLastSendAtRef = useRef(0)
  const liveLoopRef = useRef<number | null>(null)
  const predictBusyRef = useRef(false)
  const predictionTrailRef = useRef<string[]>([])
  const lastExecutionRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const lastStableNotifyRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const stableGestureLatchRef = useRef('')
  const stableHoldRef = useRef<{ label: string; since: number }>({ label: '', since: 0 })
  const lastHandSeenAtRef = useRef(0)
  const lastFrameAtRef = useRef(0)
  const selectedKeyRef = useRef(0)
  const typingTextRef = useRef('')
  const lastTypingGestureRef = useRef<{ label: string; ts: number }>({ label: '', ts: 0 })
  const cameraRunningRef = useRef(false)
  const liveTrainActiveRef = useRef(false)
  const liveTrainFramesRef = useRef<number[][][]>([])
  const liveTrainTimerRef = useRef<number | null>(null)
  const liveTrainTickerRef = useRef<number | null>(null)
  const pinchRatioRef = useRef(1)
  const pinchVelocityRef = useRef(0)
  const pinchIntentRef = useRef(false)
  const pinchConfirmedRef = useRef(false)
  const pinchStableFramesRef = useRef(0)
  const pointerLockUntilRef = useRef(0)
  const cameraOwnerIdRef = useRef(`lab-camera-${Math.random().toString(36).slice(2)}`)

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
  const [triggerCustomContext, setTriggerCustomContext] = useState('site:x.com')
  const [minConfidence, setMinConfidence] = useState(0.58)
  const [status, setStatus] = useState('Create a lightweight profile, start webcam, and train custom gestures.')
  const [mappingSaved, setMappingSaved] = useState(false)
  const [recordingState, setRecordingState] = useState<'idle' | 'recording' | 'trained'>('idle')
  const [liveTrainActive, setLiveTrainActive] = useState(false)
  const [liveTrainRemainingMs, setLiveTrainRemainingMs] = useState(0)
  const [guidedMode, setGuidedMode] = useState(true)
  const [liveStatus, setLiveStatus] = useState<'idle' | 'watching' | 'detected' | 'executing'>('idle')
  const [activityLog, setActivityLog] = useState<string[]>([])

  const [trainLabel, setTrainLabel] = useState('pinch')
  const [mapLabel, setMapLabel] = useState('pinch')
  const [mapContext, setMapContext] = useState('global')
  const [mapCustomContext, setMapCustomContext] = useState('site:x.com')
  const [mapActionType, setMapActionType] = useState<ActionType>('open_url')
  const [mapActionValue, setMapActionValue] = useState('https://www.google.com')
  const [mapCooldown, setMapCooldown] = useState(1500)
  const [mapDescription, setMapDescription] = useState('Open browser quickly')
  const [selectedActionPreset, setSelectedActionPreset] = useState('')
  const [motionTrainingPresetId, setMotionTrainingPresetId] = useState<MotionTrainingPreset['id']>('free')
  const [captureIterations, setCaptureIterations] = useState(ITERATIVE_CAPTURE_DEFAULT)
  const [captureIterationGapMs, setCaptureIterationGapMs] = useState(ITERATIVE_CAPTURE_GAP_DEFAULT_MS)
  const [iterativeCaptureRunning, setIterativeCaptureRunning] = useState(false)
  const [iterativeCaptureStep, setIterativeCaptureStep] = useState(0)
  const [captureDockMinimized, setCaptureDockMinimized] = useState(false)

  const [typingEnabled, setTypingEnabled] = useState(true)
  const [uiMode, setUiMode] = useState<'friendly' | 'advanced'>('friendly')
  const [typingText, setTypingText] = useState('')
  const [searchStatus, setSearchStatus] = useState('Gesture typing ready.')
  const [selectedKey, setSelectedKey] = useState(0)
  const [typingBindings, setTypingBindings] = useState<TypingBindings>(DEFAULT_BINDINGS)
  const [modelEngine, setModelEngine] = useState<ModelEngine>('lightweight')
  const [deepBackbone, setDeepBackbone] = useState<'resnet18' | 'resnet34'>('resnet18')
  const [deepTemporalHead, setDeepTemporalHead] = useState<'lstm' | 'bilstm_attention'>('bilstm_attention')
  const [deepStatus, setDeepStatus] = useState<DeepModelStatus | null>(null)
  const [deepTrainBusy, setDeepTrainBusy] = useState(false)
  const [editingLabel, setEditingLabel] = useState('')
  const [editingLabelDraft, setEditingLabelDraft] = useState('')

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

  const isAdvanced = uiMode === 'advanced'

  const resolvedMapContext = useMemo(
    () => normalizeContext(mapContext === 'custom' ? mapCustomContext : mapContext),
    [mapContext, mapCustomContext],
  )

  const resolvedTriggerContext = useMemo(
    () => normalizeContext(triggerContext === 'custom' ? triggerCustomContext : triggerContext),
    [triggerContext, triggerCustomContext],
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

  const mappedActions = useMemo(() => {
    if (!activeProfile) {
      return [] as Array<{
        context: string
        label: string
        actionType: ActionType
        value: string
        cooldown: number
      }>
    }

    const rows: Array<{
      context: string
      label: string
      actionType: ActionType
      value: string
      cooldown: number
    }> = []

    Object.entries(activeProfile.mappings).forEach(([contextName, contextMap]) => {
      Object.entries(contextMap).forEach(([label, action]) => {
        if (!action.enabled || action.action_type === 'none') {
          return
        }
        rows.push({
          context: contextName,
          label,
          actionType: action.action_type,
          value: action.value,
          cooldown: action.cooldown_ms,
        })
      })
    })

    return rows.sort((a, b) => `${a.context}:${a.label}`.localeCompare(`${b.context}:${b.label}`))
  }, [activeProfile])

  const deepReady = Boolean(deepStatus?.available)
  const deepRuntimeUnavailable = useMemo(
    () => /unavailable|install\s+torch|import error/i.test(deepStatus?.detail ?? ''),
    [deepStatus],
  )

  const motionTrainingPreset = useMemo(
    () => MOTION_TRAINING_PRESETS.find((item) => item.id === motionTrainingPresetId) ?? MOTION_TRAINING_PRESETS[0],
    [motionTrainingPresetId],
  )

  const savedGestureRows = useMemo(() => {
    if (!activeProfile) {
      return [] as Array<{ label: string; samples: number }>
    }

    return Object.entries(activeProfile.classes)
      .map(([label, state]) => ({ label, samples: state.samples }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [activeProfile])

  const selectedProfileSummary = useMemo(
    () => ({
      labels: activeProfile?.labels.length ?? 0,
      classes: activeProfile ? Object.keys(activeProfile.classes).length : 0,
      actions: mappedActions.length,
    }),
    [activeProfile, mappedActions.length],
  )

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
    () => stepCompletion.profileCreated && stepCompletion.trained && (!autoExecute || stepCompletion.mapped),
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
    if (!stepCompletion.trained) {
      return 'Capture at least one training sample.'
    }
    if (autoExecute && !stepCompletion.mapped) {
      return 'Save mapping or disable Auto execute.'
    }
    if (!stepCompletion.trackingReady) {
      return 'Start Profile will auto-start camera. Keep hand visible when prompted.'
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
    if (profileId) {
      void refreshDeepStatus(profileId)
    }
  }, [backendOnline])

  useEffect(() => {
    if (!backendOnline || !profileId) {
      setDeepStatus(null)
      return
    }
    void refreshDeepStatus(profileId)
  }, [backendOnline, profileId])

  useEffect(() => {
    if (modelEngine === 'deep' && !deepReady) {
      if (deepRuntimeUnavailable) {
        setStatus(`Deep engine unavailable in current backend. ${deepStatus?.detail ?? 'Use lightweight mode or install torch/torchvision.'}`)
        return
      }
      setStatus('Deep engine selected. Run Deep train model to activate ResNet + LSTM inference.')
    }
  }, [modelEngine, deepReady, deepRuntimeUnavailable, deepStatus])

  useEffect(() => {
    if (!backendOnline || !profileId) {
      return
    }

    const syncInterval = window.setInterval(() => {
      void refreshProfiles(true)
      void refreshDeepStatus(profileId, true)
    }, PROFILE_SYNC_MS)

    return () => {
      window.clearInterval(syncInterval)
    }
  }, [backendOnline, profileId, liveMode, cameraState])

  useEffect(() => {
    const onCameraRequest = (event: Event) => {
      const custom = event as CustomEvent<{ source?: string }>
      const source = custom.detail?.source ?? ''
      if (source && source !== cameraOwnerIdRef.current && cameraRunningRef.current) {
        stopCamera()
        setStatus('Camera released to another panel. Re-open webcam when needed.')
      }
    }

    window.addEventListener('dgs-camera-request', onCameraRequest as EventListener)
    return () => {
      window.removeEventListener('dgs-camera-request', onCameraRequest as EventListener)
    }
  }, [])

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden && cameraRunningRef.current) {
        stopCamera()
        setStatus('Camera auto-paused when tab was hidden. Start webcam again when ready.')
      }
    }

    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [])

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
  }, [liveMode, autoExecute, profileId, minConfidence, resolvedTriggerContext, typingEnabled, bindingKey, modelEngine, deepReady])

  useEffect(() => {
    return () => {
      stopCamera()
      if (liveLoopRef.current) {
        window.clearInterval(liveLoopRef.current)
      }
    }
  }, [])

  async function refreshProfiles(silent = false) {
    try {
      const response = await fetch('/api/light/profiles')
      if (!response.ok) {
        return
      }
      const payload = (await response.json()) as GestureProfile[]
      setProfiles(payload)
      if (!profileId) {
        if (payload.length > 0) {
          setProfileId(payload[0].id)
          setProfileLabels(payload[0].labels.join(','))
          setSequenceLength(payload[0].sequence_length)
        }
        return
      }

      const current = payload.find((item) => item.id === profileId)
      if (!current) {
        if (payload.length > 0) {
          setProfileId(payload[0].id)
          setProfileLabels(payload[0].labels.join(','))
          setSequenceLength(payload[0].sequence_length)
        }
        return
      }

      if (!silent) {
        setProfileLabels(current.labels.join(','))
        setSequenceLength(current.sequence_length)
      }
    } catch {
      // Ignore profile sync errors. UI will retry on next cycle.
    }
  }

  async function refreshDeepStatus(currentProfileId: string, silent = false) {
    try {
      const response = await fetch(`/api/deep/models/${currentProfileId}`)
      if (!response.ok) {
        if (!silent) {
          setDeepStatus(null)
        }
        return
      }

      const payload = (await response.json()) as DeepModelStatus
      setDeepStatus(payload)
    } catch {
      if (!silent) {
        setDeepStatus(null)
      }
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

  function applyActionPreset(presetId: string) {
    setSelectedActionPreset(presetId)
    const preset = ACTION_PRESETS.find((item) => item.id === presetId)
    if (!preset) {
      return
    }

    setMapActionType(preset.action_type)
    setMapActionValue(preset.value)
    setMapDescription(preset.description)
    setStatus(`Preset loaded: ${preset.name}`)
  }

  async function trainDeepModel() {
    if (!profileId) {
      setStatus('Create/select a profile first.')
      return
    }

    setDeepTrainBusy(true)
    try {
      const response = await fetch('/api/deep/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          backbone: deepBackbone,
          temporal_head: deepTemporalHead,
          epochs: 12,
          batch_size: 16,
          learning_rate: 0.001,
        }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Deep training failed' }))
        setStatus(err.detail ?? 'Deep training failed.')
        return
      }

      await refreshDeepStatus(profileId)
      setModelEngine('deep')
      setStatus('Deep model trained successfully. Engine switched to deep mode.')
      logActivity(`Deep model trained (${deepBackbone} + ${deepTemporalHead}).`)
    } finally {
      setDeepTrainBusy(false)
    }
  }

  function startLabelEdit(label: string) {
    setEditingLabel(label)
    setEditingLabelDraft(label)
  }

  async function saveLabelEdit() {
    if (!profileId || !editingLabel) {
      return
    }

    const next = normalizeLabel(editingLabelDraft)
    if (!next) {
      setStatus('New gesture label cannot be empty.')
      return
    }

    const response = await fetch('/api/light/labels/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: profileId,
        old_label: editingLabel,
        new_label: next,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Label rename failed' }))
      setStatus(err.detail ?? 'Label rename failed.')
      return
    }

    setEditingLabel('')
    setEditingLabelDraft('')
    await refreshProfiles(true)
    setTrainLabel((prev) => (prev === editingLabel ? next : prev))
    setMapLabel((prev) => (prev === editingLabel ? next : prev))
    setStatus(`Gesture label renamed: ${editingLabel} -> ${next}`)
    logActivity(`Renamed gesture ${editingLabel} to ${next}.`)
  }

  async function deleteGestureLabel(label: string) {
    if (!profileId) {
      return
    }

    const response = await fetch('/api/light/labels/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: profileId,
        label,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Could not delete gesture label' }))
      setStatus(err.detail ?? 'Could not delete gesture label.')
      return
    }

    setEditingLabel('')
    setEditingLabelDraft('')
    await refreshProfiles(true)
    setStatus(`Deleted gesture label: ${label}`)
    logActivity(`Deleted gesture ${label}.`)
  }

  function logActivity(message: string) {
    const timestamp = new Date().toLocaleTimeString()
    setActivityLog((prev) => [`${timestamp} - ${message}`, ...prev].slice(0, ACTION_LOG_LIMIT))
  }

  function clearLiveTrainTimers() {
    if (liveTrainTimerRef.current) {
      window.clearTimeout(liveTrainTimerRef.current)
      liveTrainTimerRef.current = null
    }
    if (liveTrainTickerRef.current) {
      window.clearInterval(liveTrainTickerRef.current)
      liveTrainTickerRef.current = null
    }
    liveTrainActiveRef.current = false
    setLiveTrainActive(false)
    setLiveTrainRemainingMs(0)
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
    if (profile.labels[0]) {
      setTrainLabel(profile.labels[0])
      setMapLabel(profile.labels[0])
    }
    setMapContext('global')
    setTriggerContext('global')
    setDeepStatus(null)
    setMappingSaved(false)
    setRecordingState('idle')
    setStatus(`Profile ${profile.name} is now selected (${profile.id}).`)
    logActivity(`Profile created and selected (${profile.id}).`)
  }

  async function waitForFreshHand(timeoutMs: number) {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      const fresh = lastHandSeenAtRef.current > 0 && Date.now() - lastHandSeenAtRef.current <= HAND_STALE_MS + 120
      if (cameraRunningRef.current && fresh) {
        return true
      }
      // eslint-disable-next-line no-await-in-loop
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 120)
      })
    }
    return false
  }

  async function handleLiveModeToggle(nextState: boolean) {
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
    if (!stepCompletion.trained) {
      setStatus('Capture more training samples before starting live mode.')
      return
    }
    if (autoExecute && !stepCompletion.mapped) {
      setStatus('Save at least one mapping, or disable Auto execute and retry Start profile.')
      return
    }

    if (cameraState !== 'streaming') {
      setStatus('Starting webcam automatically for profile live mode...')
      await startCamera()
    }

    const trackingReadyNow = handDetected || await waitForFreshHand(4200)
    if (!trackingReadyNow) {
      setStatus('Camera is on. Show your hand clearly for 1-2 seconds, then press Start profile again.')
      return
    }

    setLiveMode(true)
    setLiveStatus('watching')
    if (modelEngine === 'deep' && !deepReady) {
      setStatus('Profile live mode started. Deep model not trained yet, so lightweight prediction is active.')
      logActivity('Live mode started with lightweight fallback (deep model not trained).')
      return
    }
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
    clearLiveTrainTimers()
    if (detectLoopRef.current) {
      window.cancelAnimationFrame(detectLoopRef.current)
      detectLoopRef.current = null
    }
    detectProcessingRef.current = false
    detectLastSendAtRef.current = 0

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
    stableHoldRef.current = { label: '', since: 0 }
    lastHandSeenAtRef.current = 0
    lastFrameAtRef.current = 0
    pinchRatioRef.current = 1
    pinchVelocityRef.current = 0
    pinchIntentRef.current = false
    pinchConfirmedRef.current = false
    pinchStableFramesRef.current = 0
    pointerLockUntilRef.current = 0
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
      window.dispatchEvent(
        new CustomEvent('dgs-camera-request', {
          detail: { source: cameraOwnerIdRef.current },
        }),
      )

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
          stableHoldRef.current = { label: '', since: 0 }
          lastHandSeenAtRef.current = 0
          pinchIntentRef.current = false
          pinchVelocityRef.current = 0
          pinchConfirmedRef.current = false
          pinchStableFramesRef.current = 0
          setHandDetected(false)
          setLandmarkCount(0)
          drawLandmarks(null)
          return
        }

        lastHandSeenAtRef.current = Date.now()
        setHandDetected(true)
        setLandmarkCount(detected.length)
        let pinchLockActive = false
        if (detected[4] && detected[8]) {
          const distance3 = (a: { x: number; y: number; z?: number }, b: { x: number; y: number; z?: number }) => {
            const dx = a.x - b.x
            const dy = a.y - b.y
            const dz = (a.z ?? 0) - (b.z ?? 0)
            return Math.sqrt(dx * dx + dy * dy + dz * dz)
          }

          const thumbIndexTipDist = distance3(detected[4], detected[8])
          const thumbIndexJointDist = detected[7] ? distance3(detected[4], detected[7]) : thumbIndexTipDist
          const thumbIndexBaseDist = detected[3] ? distance3(detected[3], detected[8]) : thumbIndexTipDist
          const palmWidth = detected[5] && detected[17] ? distance3(detected[5], detected[17]) : 0.12
          const palmDepth = detected[0] && detected[9] ? distance3(detected[0], detected[9]) : palmWidth
          const palmScale = Math.max(0.045, (palmWidth + palmDepth) / 2)

          const pinchDistance = Math.min(
            thumbIndexTipDist,
            thumbIndexJointDist * 1.08,
            thumbIndexBaseDist * 1.1,
          )

          const pinchRatio = pinchDistance / palmScale
          const velocity = pinchRatio - pinchRatioRef.current
          pinchVelocityRef.current = velocity
          pinchRatioRef.current = pinchRatio

          const pinchEntering = (
            pinchRatio < PINCH_RATIO_ENTER
            || (pinchRatio < PINCH_RATIO_TRANSITION && velocity < PINCH_RATIO_VELOCITY)
          )
          if (pinchEntering) {
            pinchStableFramesRef.current += 1
          } else if (pinchRatio > PINCH_RATIO_RELEASE) {
            pinchStableFramesRef.current = 0
          }

          pinchIntentRef.current = pinchEntering
          pinchConfirmedRef.current = (
            pinchStableFramesRef.current >= PINCH_STABLE_FRAMES
            && pinchRatio < PINCH_RATIO_CONFIRM
          )

          if (pinchEntering || pinchConfirmedRef.current) {
            pointerLockUntilRef.current = Date.now() + POINTER_LOCK_MS_ON_PINCH
            pinchLockActive = true
          }
        }

        if (detected[8]) {
          const targetX = Math.max(0, Math.min(1, 1 - detected[8].x))
          const targetY = Math.max(0, Math.min(1, detected[8].y))
          setPointer((prev) => ({
            ...(() => {
              const dx = targetX - prev.x
              const dy = targetY - prev.y
              const movement = Math.hypot(dx, dy)
              if (movement < POINTER_DEADZONE) {
                return prev
              }

              const pointerLocked = pinchLockActive || Date.now() < pointerLockUntilRef.current
              if (pointerLocked && movement < 0.07) {
                return prev
              }

              const alpha = pointerLocked
                ? POINTER_LOCK_ALPHA
                : movement > 0.11
                  ? POINTER_FAST_ALPHA
                  : POINTER_SLOW_ALPHA

              return {
                x: prev.x + dx * alpha,
                y: prev.y + dy * alpha,
              }
            })(),
          }))
        }

        drawLandmarks(detected)

        const frame = detected.slice(0, 21).map((point: any) => [point.x, point.y, point.z ?? 0])
        sequenceBufferRef.current.push(frame)
        if (sequenceBufferRef.current.length > 90) {
          sequenceBufferRef.current.shift()
        }

        if (liveTrainActiveRef.current) {
          liveTrainFramesRef.current.push(frame)
        }
      })

      streamRef.current = stream
      video.srcObject = stream
      await video.play().catch(() => undefined)

      handsRef.current = hands
      cameraRunningRef.current = true

      const detectLoop = async (timestamp: number) => {
        if (!cameraRunningRef.current) {
          return
        }

        const runningVideo = videoRef.current
        const runningHands = handsRef.current
        const dueForFrame = timestamp - detectLastSendAtRef.current >= DETECT_LOOP_MS

        if (
          runningVideo
          && runningHands
          && runningVideo.readyState >= 2
          && dueForFrame
          && !detectProcessingRef.current
        ) {
          detectProcessingRef.current = true
          detectLastSendAtRef.current = timestamp
          try {
            await runningHands.send({ image: runningVideo })
          } catch {
            // ignore frame-level errors
          } finally {
            detectProcessingRef.current = false
          }
        }

        detectLoopRef.current = window.requestAnimationFrame((nextTs) => {
          void detectLoop(nextTs)
        })
      }

      detectLoopRef.current = window.requestAnimationFrame((nextTs) => {
        void detectLoop(nextTs)
      })
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

  async function requestTrainSample(sequence: number[][][], label: string) {
    const response = await fetch('/api/light/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: profileId,
        label,
        sequence,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Training failed' }))
      return {
        ok: false as const,
        detail: String(err.detail ?? 'Training failed.'),
      }
    }

    const payload = await response.json()
    return {
      ok: true as const,
      payload,
    }
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

    const result = await requestTrainSample(sequence, trainLabel)
    if (!result.ok) {
      setRecordingState('idle')
      setStatus(result.detail)
      return
    }

    const payload = result.payload
    setRecordingState('trained')
    const projectedProgress = Math.min(100, Math.round(((totalTrainedSamples + 1) / Math.max(1, trainingTargetSamples)) * 100))
    setStatus(`Captured ${trainLabel}. Samples for label: ${payload.samples}. Training progress: ${projectedProgress}%`)
    logActivity(`Trained ${trainLabel}. Progress ${projectedProgress}%`)
    await refreshProfiles(true)
  }

  async function startIterativeCaptureTraining() {
    if (!profileId) {
      setStatus('Create/select a profile first.')
      return
    }

    if (cameraState !== 'streaming') {
      setStatus('Start webcam before iterative capture training.')
      return
    }

    if (iterativeCaptureRunning || liveTrainActiveRef.current) {
      return
    }

    const rounds = Math.max(1, Math.min(ITERATIVE_CAPTURE_MAX, captureIterations))
    const gapMs = Math.max(ITERATIVE_CAPTURE_GAP_MIN_MS, captureIterationGapMs)
    let success = 0

    setIterativeCaptureRunning(true)
    setIterativeCaptureStep(1)
    setRecordingState('recording')
    logActivity(`Iterative capture started for ${trainLabel}: ${rounds} rounds.`)

    for (let step = 1; step <= rounds; step += 1) {
      if (!cameraRunningRef.current) {
        break
      }

      setIterativeCaptureStep(step)
      if (step === 1) {
        setStatus(`Iteration ${step}/${rounds} started. Perform ${trainLabel} now.`)
      } else {
        setStatus(`Sample ${step - 1} collected. Iteration ${step}/${rounds} started, do the action again.`)
      }

      const waitDuration = step === 1 ? 420 : gapMs
      // eslint-disable-next-line no-await-in-loop
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, waitDuration)
      })

      const sequence = getWindowSequence()
      if (!sequence) {
        logActivity(`Iteration ${step}/${rounds} skipped: not enough frames.`)
        continue
      }

      // eslint-disable-next-line no-await-in-loop
      const result = await requestTrainSample(sequence, trainLabel)
      if (!result.ok) {
        logActivity(`Iteration ${step}/${rounds} failed: ${result.detail}`)
        continue
      }

      success += 1
      logActivity(`Iteration ${step}/${rounds} trained ${trainLabel} (samples: ${result.payload.samples}).`)
    }

    setIterativeCaptureRunning(false)
    setIterativeCaptureStep(0)

    if (success === 0) {
      setRecordingState('idle')
      setStatus('Iterative capture finished, but no samples were stored. Keep hand visible and retry.')
      return
    }

    setRecordingState('trained')
    await refreshProfiles(true)
    setStatus(`Iterative capture complete: ${success}/${rounds} samples collected for ${trainLabel}.`)
  }

  function startLiveTrainCapture() {
    if (!profileId) {
      setStatus('Create/select a profile first.')
      return
    }
    if (cameraState !== 'streaming') {
      setStatus('Start webcam before live training.')
      return
    }
    if (liveTrainActiveRef.current || iterativeCaptureRunning) {
      return
    }

    const activeProfileId = profileId
    const activeTrainLabel = trainLabel
    const activeMotionPreset = motionTrainingPreset
    const captureDurationMs = LIVE_TRAIN_DURATION_MS + activeMotionPreset.extraDurationMs
    const captureSampleCount = activeMotionPreset.sampleCount

    clearLiveTrainTimers()
    liveTrainFramesRef.current = []
    liveTrainActiveRef.current = true
    setLiveTrainActive(true)
    setLiveTrainRemainingMs(captureDurationMs)
    setRecordingState('recording')
    setStatus(`Live training (${activeMotionPreset.name}) started for ${activeTrainLabel}. ${activeMotionPreset.instruction}`)
    logActivity(`Live training started for ${activeTrainLabel} (${activeMotionPreset.id})`)

    liveTrainTickerRef.current = window.setInterval(() => {
      setLiveTrainRemainingMs((prev) => Math.max(0, prev - 100))
    }, 100)

    liveTrainTimerRef.current = window.setTimeout(() => {
      const clip = liveTrainFramesRef.current.slice()
      clearLiveTrainTimers()

      void (async () => {
        if (clip.length < Math.max(sequenceLength, 14)) {
          setRecordingState('idle')
          setStatus('Live clip too short. Keep hand visible and retry live training.')
          return
        }

        const response = await fetch('/api/light/train_clip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            profile_id: activeProfileId,
            label: activeTrainLabel,
            clip,
            sample_count: captureSampleCount,
          }),
        })

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: 'Live training failed' }))
          setRecordingState('idle')
          setStatus(err.detail ?? 'Live training failed.')
          return
        }

        const payload = (await response.json()) as TrainClipResult
        setRecordingState('trained')
        setStatus(
          `Live training (${activeMotionPreset.name}) captured ${payload.samples_added} sequence samples for ${activeTrainLabel}. Total samples: ${payload.samples}.`,
        )
        logActivity(`Live trained ${activeTrainLabel} (+${payload.samples_added} samples).`)
        await refreshProfiles()
      })()
    }, captureDurationMs)
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

  function applyTypingGesture(label: string): boolean {
    if (!typingEnabled) {
      return false
    }

    const normalized = normalizeLabel(label)
    const now = Date.now()
    if (lastTypingGestureRef.current.label === normalized && now - lastTypingGestureRef.current.ts < 700) {
      return false
    }
    lastTypingGestureRef.current = { label: normalized, ts: now }

    if (normalized === typingBindings.nextLabel) {
      setSelectedKey((prev) => (prev + 1) % KEYBOARD_KEYS.length)
      return true
    }
    if (normalized === typingBindings.prevLabel) {
      setSelectedKey((prev) => (prev - 1 + KEYBOARD_KEYS.length) % KEYBOARD_KEYS.length)
      return true
    }
    if (normalized === typingBindings.selectLabel) {
      const key = KEYBOARD_KEYS[selectedKeyRef.current]
      if (key) {
        appendTyping(key)
      }
      return true
    }
    if (normalized === typingBindings.backspaceLabel) {
      backspaceTyping()
      return true
    }
    if (normalized === typingBindings.spaceLabel) {
      appendTyping(' ')
      return true
    }
    if (normalized === typingBindings.submitLabel) {
      executeSearch()
      return true
    }

    return false
  }

  async function runPrediction(executeMapped: boolean) {
    if (!profileId || predictBusyRef.current) {
      return
    }

    const useDeepEngine = modelEngine === 'deep' && deepReady

    const nowTime = Date.now()
    const handFresh = lastHandSeenAtRef.current > 0 && nowTime - lastHandSeenAtRef.current <= HAND_STALE_MS
    const frameFresh = lastFrameAtRef.current > 0 && nowTime - lastFrameAtRef.current <= FRAME_STALE_MS

    if (!handDetected || !handFresh || !frameFresh) {
      predictionTrailRef.current = []
      stableGestureLatchRef.current = ''
      stableHoldRef.current = { label: '', since: 0 }
      if (liveMode) {
        setLiveStatus('watching')
      }
      return
    }

    const sequence = getWindowSequence()
    if (!sequence) {
      predictionTrailRef.current = []
      stableGestureLatchRef.current = ''
      stableHoldRef.current = { label: '', since: 0 }
      if (liveMode) {
        setLiveStatus('watching')
      }
      return
    }

    predictBusyRef.current = true
    try {
      const response = await fetch(useDeepEngine ? '/api/deep/predict' : '/api/light/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          sequence,
          context: resolvedTriggerContext,
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
        stableHoldRef.current = { label: '', since: 0 }
        if (liveMode) {
          setLiveStatus('watching')
        }
        return
      }

      const normalized = normalizeLabel(prediction.label)
      predictionTrailRef.current = [...predictionTrailRef.current.slice(-(STABILITY_WINDOW - 1)), normalized]
      const stableVotes = predictionTrailRef.current.filter((item) => item === normalized).length
      const requiredVotes = normalized === 'pinch'
        ? STABILITY_REQUIRED
        : Math.min(STABILITY_WINDOW, STABILITY_REQUIRED + NON_PINCH_STABILITY_EXTRA)
      const stableAccepted = stableVotes >= requiredVotes

      if (!stableAccepted) {
        stableHoldRef.current = { label: '', since: 0 }
        if (liveMode) {
          setLiveStatus('detected')
        }
        return
      }

      const now = Date.now()

      if (stableHoldRef.current.label !== normalized) {
        stableHoldRef.current = { label: normalized, since: now }
      }

      const holdMs = now - stableHoldRef.current.since
      if (holdMs < EXECUTE_HOLD_MS) {
        if (liveMode) {
          setLiveStatus('detected')
        }
        return
      }

      const pinchIntent = pinchIntentRef.current
      const pinchConfirmed = pinchConfirmedRef.current

      const requiredConfidence = normalized === 'pinch'
        ? Math.min(0.97, minConfidence + EXECUTE_CONFIDENCE_MARGIN_PINCH)
        : Math.min(0.99, minConfidence + EXECUTE_CONFIDENCE_MARGIN_NON_PINCH)
      const pinchConfidenceAssist = pinchIntent
        && prediction.confidence >= Math.min(0.99, requiredConfidence + PINCH_INTENT_CONFIDENCE_ASSIST)

      if (pinchIntent && normalized !== 'pinch') {
        setStatus(`Pinch transition detected. Suppressing ${normalized} to avoid wrong action.`)
        return
      }

      if (normalized === 'pinch' && !(pinchConfirmed || pinchConfidenceAssist)) {
        setStatus('Pinch intent seen. Waiting for full pinch close...')
        return
      }

      if (prediction.confidence < requiredConfidence) {
        setStatus(
          `Detected ${normalized}, confidence too low for action (${(prediction.confidence * 100).toFixed(1)}% < ${(requiredConfidence * 100).toFixed(1)}%).`,
        )
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

      const typingConsumed = applyTypingGesture(normalized)

      if (typingConsumed) {
        setStatus(`Typing control: ${normalized}. Mapped action paused for this gesture.`)
        return
      }

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
          context: resolvedTriggerContext,
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
        context: resolvedMapContext,
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
        <p className="eyebrow">Dynamic trainer (lightweight + deep)</p>
        <h3>Train custom gestures from webcam using prototype mode or ResNet + LSTM mode</h3>
      </div>
      <div className="workflow-shell">
        <div className="workflow-head">
          <div>
            <p className="eyebrow">Guided workflow</p>
            <h4>Create/select profile, train static + motion gestures, map actions, then start live profile</h4>
          </div>
          <div className="workflow-controls">
            <span className={`engine-pill ${liveStatus}`}>Engine: {liveStatus.toUpperCase()}</span>
            <div className="mode-toggle-group" role="tablist" aria-label="UI mode">
              <button
                type="button"
                className={`mode-chip ${uiMode === 'friendly' ? 'active' : ''}`}
                onClick={() => setUiMode('friendly')}
              >
                Friendly
              </button>
              <button
                type="button"
                className={`mode-chip ${uiMode === 'advanced' ? 'active' : ''}`}
                onClick={() => setUiMode('advanced')}
              >
                Advanced
              </button>
            </div>
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

        <div className="selected-profile-strip">
          <div>
            <span>Current profile</span>
            <strong>{activeProfile ? `${activeProfile.name} (${activeProfile.id})` : 'No profile selected'}</strong>
          </div>
          <div>
            <span>Labels</span>
            <strong>{selectedProfileSummary.labels}</strong>
          </div>
          <div>
            <span>Trained classes</span>
            <strong>{selectedProfileSummary.classes}</strong>
          </div>
          <div>
            <span>Mapped actions</span>
            <strong>{selectedProfileSummary.actions}</strong>
          </div>
        </div>

        <div className="ops-summary-grid">
          <article className="ops-summary-card">
            <p className="eyebrow">What Is Running</p>
            <ul className="ops-list">
              <li>UI mode: {uiMode}</li>
              <li>Prediction engine: {modelEngine}{modelEngine === 'deep' && !deepReady ? ' (not trained yet)' : ''}</li>
              <li>Camera: {cameraState}</li>
              <li>Hand tracking: {handDetected ? 'active' : 'not detected'}</li>
              <li>Profile live: {liveMode ? 'on' : 'off'}</li>
              <li>Execution context: {resolvedTriggerContext}</li>
              <li>Deep model: {deepReady ? `${deepStatus?.backbone ?? 'resnet'} + ${deepStatus?.temporal_head ?? 'lstm'}` : 'not trained'}</li>
              <li>Last prediction: {lastPrediction?.label ? `${lastPrediction.label} (${(lastPrediction.confidence * 100).toFixed(1)}%)` : 'none'}</li>
            </ul>
          </article>

          <article className="ops-summary-card">
            <p className="eyebrow">Mapped Actions</p>
            {mappedActions.length === 0 ? (
              <p className="activity-empty">No enabled mappings yet.</p>
            ) : (
              <ul className="ops-list compact">
                {mappedActions.slice(0, 7).map((item) => (
                  <li key={`${item.context}-${item.label}-${item.actionType}`}>
                    [{item.context}] {item.label} {'->'} {item.actionType} ({item.value || 'no value'})
                  </li>
                ))}
              </ul>
            )}
          </article>
        </div>
      </div>

      <aside className={`capture-quick-dock ${captureDockMinimized ? 'minimized' : ''}`} aria-label="Quick capture controls">
        <div className="capture-quick-head">
          <p className="eyebrow">Quick capture</p>
          <button
            type="button"
            className="mini-toggle"
            onClick={() => setCaptureDockMinimized((prev) => !prev)}
          >
            {captureDockMinimized ? 'Expand' : 'Minimize'}
          </button>
        </div>

        {!captureDockMinimized && (
          <>
            <div className="capture-quick-actions">
              <button
                type="button"
                className="primary-button"
                onClick={cameraState === 'streaming' ? stopCamera : () => void startCamera()}
              >
                {cameraState === 'streaming' ? 'Stop webcam' : 'Start webcam'}
              </button>

              <button
                type="button"
                className="secondary-button"
                onClick={() => void captureTrainSample()}
                disabled={!profileId || cameraState !== 'streaming' || iterativeCaptureRunning || liveTrainActive}
              >
                Capture + train
              </button>

              <button
                type="button"
                className="secondary-button"
                onClick={() => void startIterativeCaptureTraining()}
                disabled={!profileId || cameraState !== 'streaming' || iterativeCaptureRunning || liveTrainActive}
              >
                {iterativeCaptureRunning
                  ? `Iteration ${iterativeCaptureStep}/${Math.max(1, captureIterations)}`
                  : 'Run iterative capture'}
              </button>

              <button
                type="button"
                className="secondary-button"
                onClick={startLiveTrainCapture}
                disabled={!profileId || cameraState !== 'streaming' || liveTrainActive || iterativeCaptureRunning}
              >
                {liveTrainActive ? 'Live training...' : 'Live train video'}
              </button>

              <button
                type="button"
                className="ghost-button"
                onClick={() => void runPrediction(false)}
                disabled={!profileId || cameraState !== 'streaming'}
              >
                Predict once
              </button>
            </div>

            <div className="capture-quick-meta">
              <span>Train label: {trainLabel || 'none'}</span>
              <span>Camera: {cameraState}</span>
              <span>
                {iterativeCaptureRunning
                  ? `Iteration ${iterativeCaptureStep}/${Math.max(1, captureIterations)} running`
                  : `Iterations: ${captureIterations}`}
              </span>
            </div>
          </>
        )}
      </aside>

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
                  <span>Profile name <InfoTip text="Friendly display name for this gesture profile." /></span>
                  <input
                    className="text-input"
                    aria-label="Profile name"
                    value={profileName}
                    onChange={(e) => setProfileName(e.target.value)}
                  />
                </label>

                <label className="input-group wide">
                  <span>Gesture labels <InfoTip text="Comma-separated labels. Include static and motion labels, for example: open,pinch,swipe_down." /></span>
                  <input
                    className="text-input"
                    aria-label="Gesture labels"
                    value={profileLabels}
                    onChange={(e) => setProfileLabels(e.target.value)}
                    placeholder="pinch,peace,three,open,fist"
                  />
                </label>

                <label className="input-group">
                  <span>Sequence length <InfoTip text="Higher values keep longer motion context. Use 24-32 for moving gestures." /></span>
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
                  <span>Active profile <InfoTip text="Selected profile drives training labels, mappings, and live execution." /></span>
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

              <div className="profile-summary">
                <span>Saved gestures</span>
                <strong>{savedGestureRows.length}</strong>
              </div>

              {savedGestureRows.length === 0 ? (
                <p className="mini-hint-line">No trained gestures yet. Capture or live-train gestures first.</p>
              ) : (
                <div className="saved-gesture-shell">
                  <ul className="ops-list compact">
                    {savedGestureRows.map((item) => (
                      <li key={item.label}>
                        {editingLabel === item.label ? (
                          <div className="action-row wide">
                            <input
                              className="text-input"
                              value={editingLabelDraft}
                              onChange={(e) => setEditingLabelDraft(e.target.value)}
                            />
                            <button type="button" className="secondary-button" onClick={() => void saveLabelEdit()}>
                              Save
                            </button>
                            <button
                              type="button"
                              className="ghost-button"
                              onClick={() => {
                                setEditingLabel('')
                                setEditingLabelDraft('')
                              }}
                            >
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div className="action-row wide">
                            <span>{item.label} ({item.samples} samples)</span>
                            <button type="button" className="secondary-button" onClick={() => startLabelEdit(item.label)}>
                              Edit
                            </button>
                            <button type="button" className="ghost-button" onClick={() => void deleteGestureLabel(item.label)}>
                              Delete
                            </button>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
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
              {isAdvanced && (
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
              )}

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
                {liveTrainActive
                  ? `Live video recording... ${Math.ceil(liveTrainRemainingMs / 1000)}s left`
                  : recordingState === 'recording'
                    ? 'Recording sample...'
                    : recordingState === 'trained'
                      ? `Trained samples: ${totalTrainedSamples}`
                      : 'Press Capture + train to start recording samples.'}
              </p>

              <p className="mini-hint-line">
                Use Live train (video) for dynamic gestures like pinch transitions. It captures multiple sequence windows from one clip.
              </p>

              <div className="form-grid compact motion-train-grid">
                <label className="input-group wide">
                  <span>Motion training mode <InfoTip text="Directional presets help train moving gestures like swipe-down and swipe-up." /></span>
                  <select
                    className="text-input"
                    value={motionTrainingPresetId}
                    onChange={(e) => setMotionTrainingPresetId(e.target.value as MotionTrainingPreset['id'])}
                  >
                    {MOTION_TRAINING_PRESETS.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.name}
                      </option>
                    ))}
                  </select>
                  <small>{motionTrainingPreset.instruction}</small>
                </label>
              </div>

              <div className="action-row">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={startLiveTrainCapture}
                  disabled={cameraState !== 'streaming' || liveTrainActive}
                >
                  {liveTrainActive ? 'Live training...' : 'Live train (video)'}
                </button>
                <span className="mini-hint-line">
                  {recordingState === 'recording'
                    ? 'Recording sample...'
                    : recordingState === 'trained'
                      ? `Trained samples: ${totalTrainedSamples}`
                      : 'Press Capture + train to start recording samples.'}
                </span>
              </div>

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
                  <span>Train label <InfoTip text="Choose which gesture label this captured sample should reinforce." /></span>
                  <select className="text-input" value={trainLabel} onChange={(e) => setTrainLabel(e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Capture iterations <InfoTip text="Runs repeat training rounds. You perform the same gesture each round for stronger accuracy." /></span>
                  <input
                    className="text-input"
                    type="number"
                    min={1}
                    max={ITERATIVE_CAPTURE_MAX}
                    value={captureIterations}
                    onChange={(e) => {
                      const next = Number(e.target.value)
                      setCaptureIterations(Number.isFinite(next) ? Math.max(1, Math.min(ITERATIVE_CAPTURE_MAX, next)) : 1)
                    }}
                  />
                  <small>Flow example: sample collected, iteration 2 started, do the action again.</small>
                </label>

                {isAdvanced && (
                  <label className="input-group">
                    <span>Iteration gap ms <InfoTip text="Delay between rounds so you can reset hand pose before next capture." /></span>
                    <input
                      className="text-input"
                      type="number"
                      min={ITERATIVE_CAPTURE_GAP_MIN_MS}
                      max={4000}
                      step={50}
                      value={captureIterationGapMs}
                      onChange={(e) => {
                        const next = Number(e.target.value)
                        setCaptureIterationGapMs(Number.isFinite(next) ? Math.max(ITERATIVE_CAPTURE_GAP_MIN_MS, Math.min(4000, next)) : ITERATIVE_CAPTURE_GAP_DEFAULT_MS)
                      }}
                    />
                  </label>
                )}

                {isAdvanced && (
                  <label className="input-group">
                    <span>Confidence threshold <InfoTip text="Higher threshold reduces accidental triggers but may require more training samples." /></span>
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
                )}

                <div className="action-row wide">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void captureTrainSample()}
                    disabled={iterativeCaptureRunning || liveTrainActive}
                  >
                    Capture + train
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void startIterativeCaptureTraining()}
                    disabled={iterativeCaptureRunning || liveTrainActive}
                  >
                    {iterativeCaptureRunning
                      ? `Iteration ${iterativeCaptureStep}/${Math.max(1, captureIterations)}`
                      : 'Capture iterative rounds'}
                  </button>
                  {isAdvanced && (
                    <button type="button" className="secondary-button" onClick={() => void runPrediction(false)}>
                      Predict once
                    </button>
                  )}
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
                  <span>Context <InfoTip text="Choose where this mapping applies: global, browser, presentation, or a specific site." /></span>
                  <select className="text-input" value={mapContext} onChange={(e) => setMapContext(e.target.value)}>
                    <option value="global">global</option>
                    <option value="browser">browser</option>
                    <option value="presentation">presentation</option>
                    <option value="custom">custom site</option>
                  </select>
                </label>

                {mapContext === 'custom' && (
                  <label className="input-group">
                    <span>Custom context (site:domain)</span>
                    <input
                      className="text-input"
                      value={mapCustomContext}
                      onChange={(e) => setMapCustomContext(e.target.value)}
                      placeholder="site:x.com"
                    />
                  </label>
                )}

                <label className="input-group">
                  <span>Gesture label <InfoTip text="Only this label triggers the action in the selected context." /></span>
                  <select className="text-input" value={mapLabel} onChange={(e) => setMapLabel(e.target.value)}>
                    {labelOptions.map((label) => (
                      <option key={label} value={label}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Action type <InfoTip text="Choose output behavior: open URL, launch app, hotkey, or type text." /></span>
                  <select className="text-input" value={mapActionType} onChange={(e) => setMapActionType(e.target.value as ActionType)}>
                    <option value="none">none</option>
                    <option value="open_url">open_url</option>
                    <option value="open_app">open_app</option>
                    <option value="hotkey">hotkey</option>
                    <option value="type_text">type_text</option>
                  </select>
                </label>

                <label className="input-group">
                  <span>Quick preset actions</span>
                  <select
                    className="text-input"
                    value={selectedActionPreset}
                    onChange={(e) => applyActionPreset(e.target.value)}
                  >
                    <option value="">Custom action</option>
                    {ACTION_PRESETS.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.name}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="input-group">
                  <span>Value</span>
                  <input className="text-input" value={mapActionValue} onChange={(e) => setMapActionValue(e.target.value)} />
                </label>

                <label className="input-group">
                  <span>Cooldown ms <InfoTip text="Prevents repeated accidental actions while you hold a gesture." /></span>
                  <input
                    className="text-input"
                    type="number"
                    min={100}
                    max={60000}
                    value={mapCooldown}
                    onChange={(e) => setMapCooldown(Number(e.target.value))}
                  />
                </label>

                {isAdvanced && (
                  <label className="input-group">
                    <span>Description</span>
                    <input className="text-input" value={mapDescription} onChange={(e) => setMapDescription(e.target.value)} />
                  </label>
                )}

                <div className="action-row wide">
                  <button type="button" className="secondary-button" onClick={() => void saveMapping()}>
                    Save mapping
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void runPrediction(true)}>
                    Predict + execute once
                  </button>
                </div>
                <p className="mini-hint-line">Resolved mapping context: {resolvedMapContext}</p>
                <p className="mini-hint-line">Hotkey examples: alt+f4, alt+tab, ctrl+shift+tab, win+down</p>
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

              <div className="form-grid compact">
                <label className="input-group">
                  <span>Prediction engine</span>
                  <select className="text-input" value={modelEngine} onChange={(e) => setModelEngine(e.target.value as ModelEngine)}>
                    <option value="lightweight">lightweight (prototype)</option>
                    <option value="deep">deep (ResNet + LSTM)</option>
                  </select>
                </label>

                <label className="input-group">
                  <span>Deep backbone</span>
                  <select className="text-input" value={deepBackbone} onChange={(e) => setDeepBackbone(e.target.value as 'resnet18' | 'resnet34')}>
                    <option value="resnet18">resnet18</option>
                    <option value="resnet34">resnet34</option>
                  </select>
                </label>

                <label className="input-group">
                  <span>Temporal head</span>
                  <select className="text-input" value={deepTemporalHead} onChange={(e) => setDeepTemporalHead(e.target.value as 'lstm' | 'bilstm_attention')}>
                    <option value="lstm">lstm</option>
                    <option value="bilstm_attention">bilstm_attention</option>
                  </select>
                </label>

                <div className="action-row wide">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void trainDeepModel()}
                    disabled={deepTrainBusy || !profileId || deepRuntimeUnavailable}
                  >
                    {deepTrainBusy ? 'Training deep model...' : 'Deep train model'}
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={!profileId}
                    onClick={() => {
                      if (profileId) {
                        void refreshDeepStatus(profileId)
                      }
                    }}
                  >
                    Refresh deep status
                  </button>
                </div>
                <p className="mini-hint-line">
                  Deep model status: {deepReady ? `ready (${deepStatus?.backbone} + ${deepStatus?.temporal_head})` : 'not trained'}
                </p>
                <p className="mini-hint-line">
                  Deep runtime: {deepStatus?.detail || 'Deep mode is optional. Lightweight mode is primary and runs without heavy ML dependencies.'}
                </p>
              </div>

              <div className="start-profile-row">
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => void handleLiveModeToggle(true)}
                  disabled={!canStartProfile || liveMode}
                >
                  {liveMode ? 'Profile running' : 'Start profile'}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => void handleLiveModeToggle(false)}
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
                  <option value="custom">custom site</option>
                </select>
              </label>

              {triggerContext === 'custom' && (
                <label className="input-group">
                  <span>Live custom context (site:domain)</span>
                  <input
                    className="text-input"
                    value={triggerCustomContext}
                    onChange={(e) => setTriggerCustomContext(e.target.value)}
                    placeholder="site:x.com"
                  />
                </label>
              )}

              <p className="mini-hint-line">Resolved live context: {resolvedTriggerContext}</p>
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
            isAdvanced ? (
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
            ) : (
              <div className="friendly-pane">
                <p className="mini-hint-line">
                  Friendly mode hides advanced typing bindings. Switch to Advanced mode to edit gesture-to-key controls.
                </p>
                <label className="input-group typing-input-group">
                  <span>Quick query / URL</span>
                  <textarea
                    className="typing-field"
                    value={typingText}
                    onChange={(e) => setTypingText(e.target.value)}
                    rows={2}
                    placeholder="Type and run quickly"
                  />
                </label>
                <div className="quick-actions">
                  <button type="button" className="secondary-button" onClick={() => executeSearch()}>
                    Open or search
                  </button>
                  <button type="button" className="secondary-button" onClick={() => setTypingText('')}>
                    Clear
                  </button>
                </div>
              </div>
            )
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
