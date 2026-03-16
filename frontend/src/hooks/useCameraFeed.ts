import { useCallback, useEffect, useRef, useState } from 'react'

type CameraState = 'idle' | 'starting' | 'streaming' | 'error'

export function useCameraFeed() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [state, setState] = useState<CameraState>('idle')
  const [error, setError] = useState<string>('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return
    }
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices()
      const cameraDevices = allDevices.filter((item) => item.kind === 'videoinput')
      setDevices(cameraDevices)
      if (!selectedDeviceId && cameraDevices[0]) {
        setSelectedDeviceId(cameraDevices[0].deviceId)
      }
    } catch {
      // Ignore device refresh errors; browser may restrict before permissions.
    }
  }, [selectedDeviceId])

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setState('idle')
  }, [])

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera APIs are not available in this browser.')
      setState('error')
      return
    }

    setState('starting')
    setError('')

    try {
      stop()

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
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      })
      attempts.push({ video: true, audio: false })

      let stream: MediaStream | null = null
      let lastError: unknown = null

      for (const constraints of attempts) {
        try {
          // eslint-disable-next-line no-await-in-loop
          stream = await navigator.mediaDevices.getUserMedia(constraints)
          break
        } catch (cameraError) {
          lastError = cameraError
        }
      }

      if (!stream) {
        throw lastError ?? new Error('Unable to start camera.')
      }

      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }

      setState('streaming')
      await refreshDevices()
    } catch (cameraError) {
      const message = cameraError instanceof Error ? cameraError.message : 'Unable to start the camera.'
      setError(
        /requested device not found|notfounderror|device/i.test(message)
          ? 'Selected camera was not found. Try another camera from the list and retry.'
          : message,
      )
      setState('error')
    }
  }, [refreshDevices, selectedDeviceId, stop])

  useEffect(() => {
    void refreshDevices()
  }, [refreshDevices])

  useEffect(() => stop, [stop])

  return {
    videoRef,
    streamRef,
    state,
    error,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshDevices,
    isStreaming: state === 'streaming',
    start,
    stop,
  }
}
