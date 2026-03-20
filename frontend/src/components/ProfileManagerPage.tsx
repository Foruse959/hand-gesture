import { useEffect, useMemo, useState } from 'react'

type ActionType = 'none' | 'open_url' | 'open_app' | 'hotkey' | 'type_text'

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

type MappingRow = {
  context: string
  label: string
  action: GestureAction
}

type ProfileManagerPageProps = {
  backendOnline: boolean
}

const ACTION_TYPES: ActionType[] = ['open_url', 'open_app', 'hotkey', 'type_text']
const CONTEXT_OPTIONS = ['global', 'browser', 'presentation']

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

export function ProfileManagerPage({ backendOnline }: ProfileManagerPageProps) {
  const [profiles, setProfiles] = useState<GestureProfile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState('')
  const [status, setStatus] = useState('Select a profile to manage labels and designed actions.')

  const [editingLabel, setEditingLabel] = useState('')
  const [editingLabelDraft, setEditingLabelDraft] = useState('')

  const [editorContext, setEditorContext] = useState('global')
  const [editorCustomContext, setEditorCustomContext] = useState('site:x.com')
  const [editorLabel, setEditorLabel] = useState('')
  const [editorActionType, setEditorActionType] = useState<ActionType>('open_url')
  const [editorValue, setEditorValue] = useState('')
  const [editorCooldown, setEditorCooldown] = useState(1500)
  const [editorDescription, setEditorDescription] = useState('')

  const activeProfile = useMemo(
    () => profiles.find((item) => item.id === selectedProfileId) ?? null,
    [profiles, selectedProfileId],
  )

  const labels = activeProfile?.labels ?? []

  const totalSamples = useMemo(() => {
    if (!activeProfile) {
      return 0
    }
    return Object.values(activeProfile.classes).reduce((total, current) => total + current.samples, 0)
  }, [activeProfile])

  const mappingRows = useMemo(() => {
    if (!activeProfile) {
      return [] as MappingRow[]
    }

    const rows: MappingRow[] = []
    Object.entries(activeProfile.mappings).forEach(([contextName, contextMap]) => {
      Object.entries(contextMap).forEach(([label, action]) => {
        if (!action.enabled || action.action_type === 'none') {
          return
        }
        rows.push({
          context: contextName,
          label,
          action,
        })
      })
    })

    return rows.sort((a, b) => `${a.context}:${a.label}`.localeCompare(`${b.context}:${b.label}`))
  }, [activeProfile])

  const resolvedEditorContext = useMemo(
    () => normalizeContext(editorContext === 'custom' ? editorCustomContext : editorContext),
    [editorContext, editorCustomContext],
  )

  useEffect(() => {
    if (!backendOnline) {
      setStatus('Backend offline. Start backend to manage profiles.')
      return
    }
    void refreshProfiles()
  }, [backendOnline])

  useEffect(() => {
    if (!backendOnline) {
      return
    }

    const syncId = window.setInterval(() => {
      void refreshProfiles(true)
    }, 5000)

    return () => {
      window.clearInterval(syncId)
    }
  }, [backendOnline])

  useEffect(() => {
    if (!activeProfile) {
      setEditorLabel('')
      return
    }

    setEditorLabel((prev) => {
      if (prev && activeProfile.labels.includes(prev)) {
        return prev
      }
      return activeProfile.labels[0] ?? ''
    })
  }, [activeProfile])

  async function refreshProfiles(silent = false) {
    try {
      const response = await fetch('/api/light/profiles')
      if (!response.ok) {
        if (!silent) {
          setStatus('Could not load profiles right now.')
        }
        return
      }

      const payload = (await response.json()) as GestureProfile[]
      setProfiles(payload)

      if (!payload.length) {
        setSelectedProfileId('')
        if (!silent) {
          setStatus('No profiles found yet. Create one in Studio.')
        }
        return
      }

      setSelectedProfileId((prev) => {
        const keepCurrent = payload.some((item) => item.id === prev)
        return keepCurrent ? prev : payload[0].id
      })

      if (!silent) {
        setStatus('Profiles refreshed.')
      }
    } catch {
      if (!silent) {
        setStatus('Profile refresh failed. Check backend connection.')
      }
    }
  }

  function beginEditLabel(label: string) {
    setEditingLabel(label)
    setEditingLabelDraft(label)
  }

  function cancelEditLabel() {
    setEditingLabel('')
    setEditingLabelDraft('')
  }

  async function saveLabelRename() {
    if (!selectedProfileId || !editingLabel) {
      return
    }

    const next = normalizeLabel(editingLabelDraft)
    if (!next) {
      setStatus('Label name cannot be empty.')
      return
    }

    const response = await fetch('/api/light/labels/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: selectedProfileId,
        old_label: editingLabel,
        new_label: next,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Rename failed.' }))
      setStatus(err.detail ?? 'Rename failed.')
      return
    }

    cancelEditLabel()
    await refreshProfiles(true)
    setEditorLabel(next)
    setStatus(`Renamed ${editingLabel} to ${next}.`)
  }

  async function deleteLabel(label: string) {
    if (!selectedProfileId) {
      return
    }

    const response = await fetch('/api/light/labels/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: selectedProfileId,
        label,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Delete failed.' }))
      setStatus(err.detail ?? 'Delete failed.')
      return
    }

    cancelEditLabel()
    await refreshProfiles(true)
    setStatus(`Deleted label ${label}.`)
  }

  function beginEditMapping(row: MappingRow) {
    const context = row.context.startsWith('site:') ? 'custom' : row.context
    setEditorContext(context)
    if (context === 'custom') {
      setEditorCustomContext(row.context)
    }
    setEditorLabel(row.label)
    setEditorActionType(row.action.action_type)
    setEditorValue(row.action.value)
    setEditorCooldown(row.action.cooldown_ms)
    setEditorDescription(row.action.description)
    setStatus(`Editing mapping for ${row.label} in ${row.context}.`)
  }

  async function saveMapping() {
    if (!selectedProfileId) {
      setStatus('Select a profile first.')
      return
    }
    if (!editorLabel) {
      setStatus('Select a gesture label first.')
      return
    }

    const response = await fetch('/api/light/mappings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: selectedProfileId,
        context: resolvedEditorContext,
        label: editorLabel,
        action_type: editorActionType,
        value: editorValue,
        enabled: true,
        cooldown_ms: editorCooldown,
        description: editorDescription,
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Could not save action mapping.' }))
      setStatus(err.detail ?? 'Could not save action mapping.')
      return
    }

    await refreshProfiles(true)
    setStatus(`Saved action for ${editorLabel} in ${resolvedEditorContext}.`)
  }

  async function deleteMapping(row: MappingRow) {
    if (!selectedProfileId) {
      return
    }

    const response = await fetch('/api/light/mappings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        profile_id: selectedProfileId,
        context: row.context,
        label: row.label,
        action_type: 'none',
        value: '',
        enabled: false,
        cooldown_ms: row.action.cooldown_ms,
        description: 'Deleted from profile manager',
      }),
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Could not delete mapping.' }))
      setStatus(err.detail ?? 'Could not delete mapping.')
      return
    }

    await refreshProfiles(true)
    setStatus(`Removed mapped action for ${row.label} in ${row.context}.`)
  }

  async function deleteAllGestureActions() {
    if (!selectedProfileId) {
      setStatus('Select a profile first.')
      return
    }

    if (mappingRows.length === 0) {
      setStatus('No gesture actions to delete for this profile.')
      return
    }

    const profileLabel = activeProfile ? `${activeProfile.name} (${activeProfile.id})` : selectedProfileId
    const responses = await Promise.all(
      mappingRows.map((row) =>
        fetch('/api/light/mappings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            profile_id: selectedProfileId,
            context: row.context,
            label: row.label,
            action_type: 'none',
            value: '',
            enabled: false,
            cooldown_ms: row.action.cooldown_ms,
            description: 'Bulk delete from profile manager',
          }),
        }),
      ),
    )

    const failedCount = responses.filter((response) => !response.ok).length
    const deletedCount = responses.length - failedCount

    await refreshProfiles(true)

    if (failedCount > 0) {
      setStatus(`Deleted ${deletedCount} gesture actions, but ${failedCount} failed for ${profileLabel}.`)
      return
    }

    setStatus(`Deleted all ${deletedCount} gesture actions for ${profileLabel}.`)
  }

  async function deleteProfile() {
    if (!activeProfile) {
      setStatus('Select a profile first.')
      return
    }

    const profileLabel = `${activeProfile.name} (${activeProfile.id})`
    const response = await fetch(`/api/light/profiles/${encodeURIComponent(activeProfile.id)}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Could not delete profile.' }))
      setStatus(err.detail ?? 'Could not delete profile.')
      return
    }

    cancelEditLabel()
    await refreshProfiles(true)
    setStatus(`Deleted profile ${profileLabel} and cleared local profile storage.`)
  }

  async function deleteAllProfiles() {
    if (profiles.length === 0) {
      setStatus('No profiles available to delete.')
      return
    }

    const confirmed = window.confirm(
      `Delete ALL ${profiles.length} profiles? This removes all profile data from this PC and cannot be undone.`,
    )
    if (!confirmed) {
      setStatus('Delete all profiles cancelled.')
      return
    }

    const profileIds = profiles.map((profile) => profile.id)
    const responses = await Promise.all(
      profileIds.map((id) =>
        fetch(`/api/light/profiles/${encodeURIComponent(id)}`, {
          method: 'DELETE',
        }),
      ),
    )

    const failedCount = responses.filter((response) => !response.ok).length
    const deletedCount = responses.length - failedCount

    cancelEditLabel()
    await refreshProfiles(true)

    if (failedCount > 0) {
      setStatus(`Deleted ${deletedCount} profiles, but ${failedCount} failed.`)
      return
    }

    setStatus(`Deleted all ${deletedCount} profiles and cleared local profile storage.`)
  }

  return (
    <section className="profiles-page-shell page-shell">
      <article className="glass-card profiles-page-head">
        <div>
          <p className="eyebrow">Profiles</p>
          <h2>Profile Management Workspace</h2>
          <p className="muted-copy">
            Manage labels and designed actions on a dedicated page without crowding camera or training controls.
          </p>
        </div>
        <div className="action-row">
          <button type="button" className="secondary-button" onClick={() => void refreshProfiles()}>
            Refresh profiles
          </button>
          <button type="button" className="danger-button" onClick={() => void deleteAllProfiles()} disabled={!profiles.length}>
            Delete all profiles
          </button>
        </div>
      </article>

      <div className="profiles-page-grid">
        <article className="glass-card profile-hub-card">
          <div className="section-heading">
            <p className="eyebrow">Profile details</p>
            <h3>Select and inspect profile</h3>
          </div>

          <label className="input-group">
            <span>Active profile</span>
            <select className="text-input" value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}>
              <option value="">Select profile</option>
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} ({profile.id})
                </option>
              ))}
            </select>
          </label>

          <div className="selected-profile-strip">
            <div>
              <span>Name</span>
              <strong>{activeProfile?.name ?? 'N/A'}</strong>
            </div>
            <div>
              <span>Sequence</span>
              <strong>{activeProfile?.sequence_length ?? 0}</strong>
            </div>
            <div>
              <span>Labels</span>
              <strong>{labels.length}</strong>
            </div>
            <div>
              <span>Samples</span>
              <strong>{totalSamples}</strong>
            </div>
          </div>

          <div className="action-row wide">
            <button
              type="button"
              className="danger-button"
              onClick={() => void deleteProfile()}
              disabled={!activeProfile}
            >
              Delete profile
            </button>
          </div>
          <p className="mini-hint-line">
            Single delete runs instantly. Only Delete all profiles asks for confirmation.
          </p>

          <div className="saved-gesture-shell profile-label-list">
            {labels.length === 0 ? (
              <p className="activity-empty">No labels found for this profile.</p>
            ) : (
              <ul className="ops-list compact">
                {labels.map((label) => {
                  const samples = activeProfile?.classes[label]?.samples ?? 0
                  return (
                    <li key={label}>
                      {editingLabel === label ? (
                        <div className="action-row wide">
                          <input
                            className="text-input"
                            value={editingLabelDraft}
                            onChange={(e) => setEditingLabelDraft(e.target.value)}
                          />
                          <button type="button" className="secondary-button" onClick={() => void saveLabelRename()}>
                            Save
                          </button>
                          <button type="button" className="ghost-button" onClick={cancelEditLabel}>
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div className="profile-hub-label-row">
                          <span>{label} ({samples} samples)</span>
                          <div className="action-row">
                            <button type="button" className="secondary-button" onClick={() => beginEditLabel(label)}>
                              Rename
                            </button>
                            <button type="button" className="ghost-button" onClick={() => void deleteLabel(label)}>
                              Delete
                            </button>
                          </div>
                        </div>
                      )}
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </article>

        <article className="glass-card profile-hub-card">
          <div className="section-heading">
            <p className="eyebrow">Designed actions</p>
            <h3>View, edit, and delete mapped actions</h3>
            <div className="action-row">
              <button
                type="button"
                className="danger-button"
                onClick={() => void deleteAllGestureActions()}
                disabled={!selectedProfileId || mappingRows.length === 0}
              >
                Delete all gesture actions
              </button>
            </div>
          </div>

          <div className="saved-gesture-shell profile-action-list">
            {mappingRows.length === 0 ? (
              <p className="activity-empty">No enabled actions in this profile yet.</p>
            ) : (
              <ul className="ops-list compact">
                {mappingRows.map((row) => (
                  <li key={`${row.context}-${row.label}-${row.action.action_type}`}>
                    <div className="profile-hub-action-row">
                      <div>
                        <strong>[{row.context}] {row.label}</strong>
                        <p className="muted-copy">{row.action.action_type} ({row.action.value || 'no value'}) | cooldown {row.action.cooldown_ms}ms</p>
                      </div>
                      <div className="action-row">
                        <button type="button" className="secondary-button" onClick={() => beginEditMapping(row)}>
                          Edit
                        </button>
                        <button type="button" className="ghost-button" onClick={() => void deleteMapping(row)}>
                          Delete
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="form-grid compact profile-action-editor">
            <label className="input-group">
              <span>Context</span>
              <select className="text-input" value={editorContext} onChange={(e) => setEditorContext(e.target.value)}>
                {CONTEXT_OPTIONS.map((contextName) => (
                  <option key={contextName} value={contextName}>{contextName}</option>
                ))}
                <option value="custom">custom site</option>
              </select>
            </label>

            {editorContext === 'custom' && (
              <label className="input-group">
                <span>Custom context (site:domain)</span>
                <input
                  className="text-input"
                  value={editorCustomContext}
                  onChange={(e) => setEditorCustomContext(e.target.value)}
                  placeholder="site:x.com"
                />
              </label>
            )}

            <label className="input-group">
              <span>Label</span>
              <select className="text-input" value={editorLabel} onChange={(e) => setEditorLabel(e.target.value)}>
                {labels.map((label) => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </select>
            </label>

            <label className="input-group">
              <span>Action type</span>
              <select className="text-input" value={editorActionType} onChange={(e) => setEditorActionType(e.target.value as ActionType)}>
                {ACTION_TYPES.map((actionType) => (
                  <option key={actionType} value={actionType}>{actionType}</option>
                ))}
              </select>
            </label>

            <label className="input-group wide">
              <span>Value</span>
              <input className="text-input" value={editorValue} onChange={(e) => setEditorValue(e.target.value)} />
            </label>

            <label className="input-group">
              <span>Cooldown ms</span>
              <input
                className="text-input"
                type="number"
                min={100}
                max={60000}
                value={editorCooldown}
                onChange={(e) => setEditorCooldown(Number(e.target.value))}
              />
            </label>

            <label className="input-group">
              <span>Description</span>
              <input
                className="text-input"
                value={editorDescription}
                onChange={(e) => setEditorDescription(e.target.value)}
              />
            </label>

            <div className="action-row wide">
              <button type="button" className="primary-button" onClick={() => void saveMapping()}>
                Save action
              </button>
            </div>
            <p className="mini-hint-line">Resolved context: {resolvedEditorContext}</p>
          </div>
        </article>
      </div>

      <p className="status-line profile-hub-status">{status}</p>
    </section>
  )
}
