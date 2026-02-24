import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type GameMode = 'human_vs_ai' | 'ai_vs_ai'
type AgentType = 'human' | 'baseline' | 'belief' | 'heuristic'

type AgentConfig = {
  type: AgentType
  checkpoint_path: string | null
  simulations: number
  device: string
  ablate_belief_head: boolean
  heuristic_bot_name: string
  heuristic_bot_epsilon: number
}

type LegalAction = {
  action_id: number
  macro: string
  position: number
  label: string
}

type PublicAction = {
  step_index: number
  round_index: number
  actor: number
  action_type: string
  source: number
  target_pos: number
  a_value: number
  b_value: number
  note: string
}

type GameState = {
  terminated: boolean
  phase: string
  current_player: number
  observer_player: number
  scores: number[]
  round_scores: number[]
  deck_size: number
  discard_top: number
  discard_size: number
  legal_actions: LegalAction[]
  board_views: Record<string, { values: number[]; visible_mask: number[] }>
  public_history: PublicAction[]
}

type SessionResponse = {
  state: Record<string, unknown>
  view: GameState
}

type AgentDecisionContext = {
  actor_player: number
  decision_phase_id: number
  pending_source: string | null
  pending_drawn_value: number | null
  pending_keep_drawn: boolean | null
}

type ActionResponse = SessionResponse & {
  ai_action?: {
    actor: number
    action_id: number
  } | null
}

type AgentStepResponse = SessionResponse & {
  actor_player: number
  decision_context: AgentDecisionContext | null
  step_log: string
  turn_completed: boolean
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
type HumanSource = 'deck' | 'discard'
type DeckDecision = 'keep' | 'discard'
const ACTION_HISTORY_UNKNOWN = -127
const UNKNOWN_VALUE = -99
const HEURISTIC_BOT_OPTIONS = [
  'greedy_value_replacement',
  'information_first_flip',
  'column_hunter',
  'risk_aware_unknown_replacement',
  'end_round_aggro',
  'anti_discard',
] as const

function cardTone(value: number): { bg: string; fg: string } {
  if (value <= -1) return { bg: '#1f4e8c', fg: '#ffffff' }
  if (value === 0) return { bg: '#9fd2ff', fg: '#14233b' }
  if (value <= 4) return { bg: '#5bbf6a', fg: '#0b2a10' }
  if (value <= 8) return { bg: '#ffd85a', fg: '#332600' }
  return { bg: '#ef6a6a', fg: '#3a0b0b' }
}

function CardBox({ label, value, unknownLabel = '?' }: { label: string; value: number | null; unknownLabel?: string }) {
  const known = typeof value === 'number'
  const toneStyle = known ? { backgroundColor: cardTone(value).bg, color: cardTone(value).fg } : undefined
  return (
    <div className={`card-box ${known ? '' : 'card-unknown'}`} style={toneStyle}>
      <small>{label}</small>
      <strong>{known ? value : unknownLabel}</strong>
    </div>
  )
}

function valueLabel(value: number): string {
  if (value === ACTION_HISTORY_UNKNOWN || value === UNKNOWN_VALUE) return 'Unknown'
  return String(value)
}

function formatActionSteps(entry: PublicAction, actorLabel: string): string[] {
  const pos = entry.target_pos + 1
  if (entry.action_type === 'SETUP_FLIP') {
    return [`${actorLabel} flipped card at ${pos} (revealed ${valueLabel(entry.a_value)})`]
  }
  if (entry.action_type === 'TAKE_DISCARD_AND_REPLACE') {
    return [
      `${actorLabel} drew from discard`,
      `${actorLabel} replaced card at ${pos} (was ${valueLabel(entry.b_value)}, now ${valueLabel(entry.a_value)})`,
    ]
  }
  if (entry.action_type === 'DRAW_DECK_KEEP_AND_REPLACE') {
    return [
      `${actorLabel} drew from deck`,
      `${actorLabel} kept drawn card`,
      `${actorLabel} replaced card at ${pos} (was ${valueLabel(entry.b_value)}, now ${valueLabel(entry.a_value)})`,
    ]
  }
  if (entry.action_type === 'DRAW_DECK_DISCARD_AND_FLIP') {
    return [
      `${actorLabel} drew from deck`,
      `${actorLabel} discarded drawn card (${valueLabel(entry.a_value)})`,
      `${actorLabel} flipped card at ${pos} (revealed ${valueLabel(entry.b_value)})`,
    ]
  }
  if (entry.action_type === 'COLUMN_CLEARED') {
    return [`${actorLabel} cleared a column (${entry.note || 'matched triplet'})`]
  }
  if (entry.action_type === 'ROUND_END') {
    return [`${actorLabel} ended the round (${entry.note || 'final turns resolved'})`]
  }
  return [`${actorLabel} ${entry.action_type.toLowerCase()}${entry.note ? ` (${entry.note})` : ''}`]
}

function formatScoreSummary(scores: number[], getActorLabel: (idx: number) => string): string {
  return scores.map((score, idx) => `${getActorLabel(idx)}=${score}`).join(', ')
}

function initialAgents(numPlayers: number, mode: GameMode): AgentConfig[] {
  return Array.from({ length: numPlayers }, (_, idx) => ({
    type: mode === 'human_vs_ai' && idx === 0 ? 'human' : 'baseline',
    checkpoint_path: null,
    simulations: 32,
    device: 'cpu',
    ablate_belief_head: false,
    heuristic_bot_name: 'greedy_value_replacement',
    heuristic_bot_epsilon: 0.02,
  }))
}

function App() {
  const [mode, setMode] = useState<GameMode>('human_vs_ai')
  const [numPlayers, setNumPlayers] = useState(2)
  const [seed, setSeed] = useState(0)
  const [observerPlayer, setObserverPlayer] = useState(0)
  const [agents, setAgents] = useState<AgentConfig[]>(initialAgents(2, 'human_vs_ai'))
  const [gameView, setGameView] = useState<GameState | null>(null)
  const [sessionState, setSessionState] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [autoRespondToAi, setAutoRespondToAi] = useState(true)
  const [autoplayDelayMs, setAutoplayDelayMs] = useState(600)
  const [isAutoRunning, setIsAutoRunning] = useState(false)
  const [selectedHumanSource, setSelectedHumanSource] = useState<HumanSource | null>(null)
  const [deckDecision, setDeckDecision] = useState<DeckDecision>('keep')
  const [pendingDeckDrawnValue, setPendingDeckDrawnValue] = useState<number | null>(null)
  const [discardCommitted, setDiscardCommitted] = useState(false)
  const [agentDecisionContext, setAgentDecisionContext] = useState<AgentDecisionContext | null>(null)
  const [streamedStepLines, setStreamedStepLines] = useState<Array<{ id: string; text: string }>>([])
  const streamedStepSeqRef = useRef(0)
  const gameSummaryLoggedRef = useRef(false)
  const historyCursorRef = useRef(0)
  const logHistoryRef = useRef<HTMLDivElement | null>(null)
  const autoScrollLogRef = useRef(true)

  const currentSeatType = useMemo(() => {
    if (!gameView) return null
    return agents[gameView.current_player]?.type ?? null
  }, [agents, gameView])

  const canPlayHumanAction = gameView !== null && !gameView.terminated && currentSeatType === 'human'
  const canStepAi = gameView !== null && !gameView.terminated && currentSeatType !== 'human'
  const getActorLabel = useCallback(
    (playerIdx: number): string => {
      const agentType = agents[playerIdx]?.type
      if (agentType === 'belief') return `Belief (P${playerIdx + 1})`
      if (agentType === 'baseline') return `Baseline (P${playerIdx + 1})`
      if (agentType === 'heuristic') return `Heuristic (P${playerIdx + 1})`
      if (agentType === 'human') return `Human (P${playerIdx + 1})`
      return `Player ${playerIdx + 1}`
    },
    [agents],
  )
  const legalByMacroAndPos = useMemo(() => {
    const map = new Map<string, number>()
    if (!gameView) return map
    for (const legal of gameView.legal_actions) {
      map.set(`${legal.macro}:${legal.position}`, legal.action_id)
    }
    return map
  }, [gameView])

  const hasLegalMacro = (macro: string) => {
    if (!gameView) return false
    return gameView.legal_actions.some((action) => action.macro === macro)
  }
  const canDiscardDrawn = hasLegalMacro('DRAW_DECK_DISCARD_AND_FLIP')

  const currentStepInstruction = useMemo(() => {
    if (!gameView) return ''
    if (gameView.terminated) return 'Game ended. Review final board and action log.'

    if (canPlayHumanAction) {
      if (gameView.phase === 'SETUP') {
        return 'Your setup step: click a hidden card on your board to reveal it.'
      }
      if (selectedHumanSource === null) {
        return 'Your step: choose Deck or Discard first.'
      }
      if (selectedHumanSource === 'discard') {
        return 'Your step: click a position on your board to replace with the discard card.'
      }
      if (deckDecision === 'discard') {
        return 'Your step: click a hidden position on your board to flip (drawn card has moved to discard).'
      }
      return 'Your step: click any position on your board to replace with the drawn card or click on the drawn card to discard it.'
    }

    const actor = getActorLabel(gameView.current_player)
    if (agentDecisionContext == null) {
      return `${actor} is selecting a source (deck/discard). Use Step AI or Play to continue.`
    }
    if (agentDecisionContext.decision_phase_id === 0) {
      return `${actor} is in setup reveal step (choosing a hidden card to reveal).`
    }
    if (agentDecisionContext.decision_phase_id === 1) {
      return `${actor} is choosing source (deck or discard).`
    }
    if (agentDecisionContext.decision_phase_id === 2) {
      return `${actor} is deciding to keep or discard the drawn card.`
    }
    if (agentDecisionContext.decision_phase_id === 3) {
      return `${actor} is choosing board position for flip/replace.`
    }
    return `${actor} is taking an action step.`
  }, [agentDecisionContext, canPlayHumanAction, deckDecision, gameView, getActorLabel, selectedHumanSource])

  const deckTopValue = useMemo(() => {
    const rawDeck = sessionState?.deck
    if (!Array.isArray(rawDeck) || rawDeck.length === 0) return null
    const top = rawDeck[rawDeck.length - 1]
    return typeof top === 'number' ? top : null
  }, [sessionState])

  const getHumanActionFromPosition = (position: number): number | null => {
    if (!gameView) return null
    if (gameView.phase === 'SETUP') {
      return legalByMacroAndPos.get(`SETUP_FLIP:${position}`) ?? null
    }
    if (!selectedHumanSource) return null
    if (selectedHumanSource === 'discard') {
      return legalByMacroAndPos.get(`TAKE_DISCARD_AND_REPLACE:${position}`) ?? null
    }
    if (deckDecision === 'discard') {
      return legalByMacroAndPos.get(`DRAW_DECK_DISCARD_AND_FLIP:${position}`) ?? null
    }
    return legalByMacroAndPos.get(`DRAW_DECK_KEEP_AND_REPLACE:${position}`) ?? null
  }

  const actionLogLines = useMemo(() => {
    const pendingLines: Array<{ id: string; text: string }> = []
    if (canPlayHumanAction && gameView && gameView.phase !== 'SETUP') {
      const actor = gameView.current_player
      const label = getActorLabel(actor)
      if (selectedHumanSource === 'deck') {
        pendingLines.push({ id: 'pending-draw', text: `${label} drew from deck` })
        if (deckDecision === 'discard') {
          pendingLines.push({
            id: 'pending-discard',
            text: `${label} discarded drawn card (${valueLabel(pendingDeckDrawnValue ?? UNKNOWN_VALUE)}) and will flip a hidden card`,
          })
        } else {
          pendingLines.push({ id: 'pending-keep', text: `${label} kept drawn card and will replace a card` })
        }
      } else if (selectedHumanSource === 'discard') {
        pendingLines.push({ id: 'pending-take-discard', text: `${label} drew from discard` })
        pendingLines.push({ id: 'pending-place-discard', text: `${label} will replace a card` })
      }
    }
    return [...streamedStepLines, ...pendingLines]
  }, [canPlayHumanAction, deckDecision, gameView, pendingDeckDrawnValue, selectedHumanSource, streamedStepLines, getActorLabel])

  const syncAgents = (nextNumPlayers: number, nextMode: GameMode) => {
    setAgents((prev) => {
      const next: AgentConfig[] = Array.from({ length: nextNumPlayers }, (_, idx) => {
        const existing = prev[idx]
        if (existing) {
          if (nextMode === 'human_vs_ai' && idx === 0) return { ...existing, type: 'human' }
          if (nextMode === 'ai_vs_ai' && existing.type === 'human') return { ...existing, type: 'baseline' }
          return existing
        }
        return {
          type: nextMode === 'human_vs_ai' && idx === 0 ? 'human' : 'baseline',
          checkpoint_path: null,
          simulations: 32,
          device: 'cpu',
          ablate_belief_head: false,
          heuristic_bot_name: 'greedy_value_replacement',
          heuristic_bot_epsilon: 0.02,
        }
      })
      if (nextMode === 'human_vs_ai') {
        next[0].type = 'human'
        for (let i = 1; i < next.length; i += 1) {
          if (next[i].type === 'human') next[i].type = 'baseline'
        }
      }
      if (nextMode === 'ai_vs_ai') {
        for (const agent of next) {
          if (agent.type === 'human') agent.type = 'baseline'
        }
      }
      return next
    })
  }

  const createGame = async (event: FormEvent) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/session/new`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_players: numPlayers,
          seed,
          setup_mode: 'auto',
          history_window_k: 16,
          score_limit: 100,
          observer_player: observerPlayer,
        }),
      })
      const payload = (await response.json()) as SessionResponse & { detail?: string }
      if (!response.ok) {
        throw new Error(payload.detail ?? 'Failed to create game')
      }
      setSessionState(payload.state)
      setGameView(payload.view)
      setAgentDecisionContext(null)
      setStreamedStepLines([])
      streamedStepSeqRef.current = 0
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const refreshState = async (nextObserverPlayer: number) => {
    if (!sessionState) return
    const response = await fetch(`${API_BASE}/api/session/view`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: sessionState, observer_player: nextObserverPlayer }),
    })
    const payload = (await response.json()) as SessionResponse & { detail?: string }
    if (!response.ok) {
      throw new Error(payload.detail ?? 'Failed to fetch state')
    }
    setSessionState(payload.state)
    setGameView(payload.view)
  }

  const submitHumanAction = async (actionId: number) => {
    if (!sessionState) return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/session/apply-action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: sessionState, action_id: actionId, observer_player: observerPlayer }),
      })
      const payload = (await response.json()) as ActionResponse & { detail?: string }
      if (!response.ok) {
        throw new Error(payload.detail ?? 'Action failed')
      }
      setSelectedHumanSource(null)
      setDeckDecision('keep')
      setPendingDeckDrawnValue(null)
      setDiscardCommitted(false)
      setSessionState(payload.state)
      setGameView(payload.view)
      setAgentDecisionContext(null)
      if (autoRespondToAi && mode === 'human_vs_ai') {
        let localState = payload.state
        let localView = payload.view
        let localContext: AgentDecisionContext | null = null
        for (let i = 0; i < 200; i += 1) {
          if (localView.terminated) break
          const currentAgent = agents[localView.current_player]
          if (!currentAgent || currentAgent.type === 'human') break
          const stepResponse = await fetch(`${API_BASE}/api/session/infer-agent-step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              state: localState,
              observer_player: observerPlayer,
              agent: currentAgent,
              decision_context: localContext,
            }),
          })
          const stepPayload = (await stepResponse.json()) as AgentStepResponse & { detail?: string }
          if (!stepResponse.ok) {
            throw new Error(stepPayload.detail ?? 'AI response failed')
          }
          streamedStepSeqRef.current += 1
          setStreamedStepLines((prev) => [
            ...prev,
            {
              id: `ai-step-${streamedStepSeqRef.current}`,
              text: `${getActorLabel(stepPayload.actor_player)} ${stepPayload.step_log}`,
            },
          ])
          localState = stepPayload.state
          localView = stepPayload.view
          localContext = stepPayload.turn_completed ? null : stepPayload.decision_context
          if (stepPayload.turn_completed && localView.current_player === 0) {
            break
          }
        }
        setSessionState(localState)
        setGameView(localView)
        setAgentDecisionContext(localContext)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const stepAiTurn = async () => {
    if (!sessionState || !gameView) return
    const currentAgent = agents[gameView.current_player]
    if (!currentAgent || currentAgent.type === 'human') return
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/session/infer-agent-step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state: sessionState,
          observer_player: observerPlayer,
          agent: currentAgent,
          decision_context: agentDecisionContext,
        }),
      })
      const payload = (await response.json()) as AgentStepResponse & { detail?: string }
      if (!response.ok) {
        throw new Error(payload.detail ?? 'AI step failed')
      }
      streamedStepSeqRef.current += 1
      setStreamedStepLines((prev) => [
        ...prev,
        {
          id: `ai-step-${streamedStepSeqRef.current}`,
          text: `${getActorLabel(payload.actor_player)} ${payload.step_log}`,
        },
      ])
      setSessionState(payload.state)
      setGameView(payload.view)
      setAgentDecisionContext(payload.turn_completed ? null : payload.decision_context)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const runAutoplayBurst = async (steps: number) => {
    if (!sessionState || !gameView) return
    setLoading(true)
    setError('')
    try {
      const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms))
      let localState = sessionState
      let localView = gameView
      let localContext = agentDecisionContext
      for (let i = 0; i < steps; i += 1) {
        if (localView.terminated) break
        const currentAgent = agents[localView.current_player]
        if (!currentAgent || currentAgent.type === 'human') break
        const response = await fetch(`${API_BASE}/api/session/infer-agent-step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            state: localState,
            observer_player: observerPlayer,
            agent: currentAgent,
            decision_context: localContext,
          }),
        })
        const payload = (await response.json()) as AgentStepResponse & { detail?: string }
        if (!response.ok) {
          throw new Error(payload.detail ?? 'Autoplay failed')
        }
        streamedStepSeqRef.current += 1
        setStreamedStepLines((prev) => [
          ...prev,
          {
            id: `ai-step-${streamedStepSeqRef.current}`,
            text: `${getActorLabel(payload.actor_player)} ${payload.step_log}`,
          },
        ])
        localState = payload.state
        localView = payload.view
        localContext = payload.turn_completed ? null : payload.decision_context

        // Keep autoplay pacing uniform across all step transitions,
        // including end-of-turn -> next player's first decision step.
        if (i < steps - 1) {
          await sleep(Math.max(100, autoplayDelayMs))
        }
      }
      setSessionState(localState)
      setGameView(localView)
      setAgentDecisionContext(localContext)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!isAutoRunning || !gameView || gameView.terminated) return
    const timer = window.setInterval(() => {
      if (!loading) {
        void runAutoplayBurst(1)
      }
    }, Math.max(100, autoplayDelayMs))
    return () => window.clearInterval(timer)
  }, [autoplayDelayMs, gameView, isAutoRunning, loading])

  useEffect(() => {
    if (!gameView) {
      autoScrollLogRef.current = true
      setStreamedStepLines([])
      streamedStepSeqRef.current = 0
      historyCursorRef.current = 0
      gameSummaryLoggedRef.current = false
    }
  }, [gameView])

  useEffect(() => {
    if (!gameView) return
    const history = gameView.public_history

    if (history.length < historyCursorRef.current) {
      historyCursorRef.current = 0
      gameSummaryLoggedRef.current = false
    }

    if (history.length > historyCursorRef.current) {
      const newEntries = history.slice(historyCursorRef.current)
      const appended: Array<{ id: string; text: string }> = []
      for (const entry of newEntries) {
        const steps = formatActionSteps(entry, getActorLabel(entry.actor))
        for (const step of steps) {
          streamedStepSeqRef.current += 1
          appended.push({
            id: `hist-${streamedStepSeqRef.current}`,
            text: step,
          })
        }
        if (entry.action_type === 'ROUND_END') {
          streamedStepSeqRef.current += 1
          appended.push({
            id: `summary-round-${streamedStepSeqRef.current}`,
            text: `Round Summary — Round Scores: ${formatScoreSummary(gameView.round_scores, getActorLabel)} | Total Scores: ${formatScoreSummary(
              gameView.scores,
              getActorLabel,
            )}`,
          })
        }
      }
      if (appended.length > 0) {
        setStreamedStepLines((prev) => [...prev, ...appended])
      }
      historyCursorRef.current = history.length
    }

    if (gameView.terminated && !gameSummaryLoggedRef.current) {
      let winnerScore = Number.POSITIVE_INFINITY
      for (const score of gameView.scores) {
        if (score < winnerScore) winnerScore = score
      }
      const winners: string[] = []
      gameView.scores.forEach((score, idx) => {
        if (score === winnerScore) winners.push(getActorLabel(idx))
      })
      streamedStepSeqRef.current += 1
      setStreamedStepLines((prev) => [
        ...prev,
        {
          id: `summary-game-${streamedStepSeqRef.current}`,
          text: `Game Summary — Final Scores: ${formatScoreSummary(gameView.scores, getActorLabel)} | Winner: ${winners.join(', ')}`,
        },
      ])
      gameSummaryLoggedRef.current = true
    } else if (!gameView.terminated) {
      gameSummaryLoggedRef.current = false
    }
  }, [gameView, getActorLabel])

  useEffect(() => {
    if (!autoScrollLogRef.current) return
    const el = logHistoryRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [actionLogLines.length])

  const handleLogScroll = () => {
    const el = logHistoryRef.current
    if (!el) return
    const thresholdPx = 8
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    autoScrollLogRef.current = distanceFromBottom <= thresholdPx
  }

  useEffect(() => {
    if (!canPlayHumanAction) {
      setSelectedHumanSource(null)
      setDeckDecision('keep')
      setPendingDeckDrawnValue(null)
      setDiscardCommitted(false)
      return
    }
    if (gameView?.phase === 'SETUP') {
      setSelectedHumanSource(null)
      setDeckDecision('keep')
      setPendingDeckDrawnValue(null)
      setDiscardCommitted(false)
    }
  }, [canPlayHumanAction, gameView?.phase])

  return (
    <main className="app">
      <header>
        <h1>Skyjo Web Arena</h1>
        <p>Play a trained agent or watch AI vs AI in real time.</p>
      </header>

      <section className="panel">
        <h2>Game Setup</h2>
        <form className="setup-grid" onSubmit={createGame}>
          <label>
            Mode
            <select
              value={mode}
              onChange={(event) => {
                const nextMode = event.target.value as GameMode
                setMode(nextMode)
                syncAgents(numPlayers, nextMode)
              }}
            >
              <option value="human_vs_ai">Human vs AI</option>
              <option value="ai_vs_ai">AI vs AI</option>
            </select>
          </label>
          <label>
            Players
            <input
              type="number"
              min={2}
              max={8}
              value={numPlayers}
              onChange={(event) => {
                const nextPlayers = Number(event.target.value)
                setNumPlayers(nextPlayers)
                syncAgents(nextPlayers, mode)
              }}
            />
          </label>
          <label>
            Seed
            <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? 'Starting...' : 'Start Game'}
          </button>
        </form>

        <div className="seats">
          {agents.map((agent, idx) => (
            <article className="seat-card" key={idx}>
              <h3>Seat {idx}</h3>
              <label>
                Type
                <select
                  value={agent.type}
                  onChange={(event) => {
                    const nextType = event.target.value as AgentType
                    setAgents((prev) => {
                      const clone = [...prev]
                      clone[idx] = { ...clone[idx], type: nextType }
                      return clone
                    })
                  }}
                  disabled={mode === 'human_vs_ai' && idx === 0}
                >
                  {mode === 'human_vs_ai' && idx === 0 ? null : <option value="human">Human</option>}
                  <option value="baseline">Baseline MuZero</option>
                  <option value="belief">Belief MuZero</option>
                  <option value="heuristic">Heuristic Bot</option>
                </select>
              </label>
              {agent.type === 'heuristic' ? (
                <>
                  <label>
                    Heuristic Bot
                    <select
                      value={agent.heuristic_bot_name}
                      onChange={(event) => {
                        const nextName = event.target.value
                        setAgents((prev) => {
                          const clone = [...prev]
                          clone[idx] = { ...clone[idx], heuristic_bot_name: nextName }
                          return clone
                        })
                      }}
                    >
                      {HEURISTIC_BOT_OPTIONS.map((name) => (
                        <option value={name} key={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Heuristic Epsilon
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={agent.heuristic_bot_epsilon}
                      onChange={(event) => {
                        const nextEps = Number(event.target.value)
                        setAgents((prev) => {
                          const clone = [...prev]
                          clone[idx] = { ...clone[idx], heuristic_bot_epsilon: nextEps }
                          return clone
                        })
                      }}
                    />
                  </label>
                </>
              ) : null}
              {agent.type === 'heuristic' ? null : (
              <label>
                Checkpoint (optional)
                <input
                  type="text"
                  placeholder="runs/muzero_baseline/checkpoints/checkpoint_iter_50.pt"
                  value={agent.checkpoint_path ?? ''}
                  onChange={(event) => {
                    const nextPath = event.target.value.trim()
                    setAgents((prev) => {
                      const clone = [...prev]
                      clone[idx] = { ...clone[idx], checkpoint_path: nextPath.length > 0 ? nextPath : null }
                      return clone
                    })
                  }}
                />
              </label>
              )}
              <label>
                Simulations
                <input
                  type="number"
                  min={1}
                  max={400}
                  value={agent.simulations}
                  disabled={agent.type === 'heuristic'}
                  onChange={(event) => {
                    const nextSims = Number(event.target.value)
                    setAgents((prev) => {
                      const clone = [...prev]
                      clone[idx] = { ...clone[idx], simulations: nextSims }
                      return clone
                    })
                  }}
                />
              </label>
            </article>
          ))}
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      {gameView ? (
        <>
          <section className="panel status">
            <h2>Game Status</h2>
            <p>
              <strong>Phase:</strong> {gameView.phase} | <strong>Current Player:</strong> {gameView.current_player}
            </p>
            <p>
              <strong>Scores:</strong> {gameView.scores.join(', ')} | <strong>Round:</strong> {gameView.round_scores.join(', ')}
            </p>
            <label>
              Observer
              <input
                type="number"
                min={0}
                max={numPlayers - 1}
                value={observerPlayer}
                onChange={async (event) => {
                  const nextObserver = Number(event.target.value)
                  setObserverPlayer(nextObserver)
                  try {
                    await refreshState(nextObserver)
                  } catch (err) {
                    setError(err instanceof Error ? err.message : 'Failed to refresh observer state')
                  }
                }}
              />
            </label>
          </section>

          <section className="panel board-section">
            <h2>Board Views</h2>
            <p className="board-instruction">{currentStepInstruction}</p>
            <div className="boards">
              {Object.entries(gameView.board_views).map(([player, board]) => (
                <article className="board" key={player}>
                  <h3>Player {player}</h3>
                  <div className="grid">
                    {board.values.map((value, idx) => {
                      const visible = board.visible_mask[idx] === 1
                      const isCurrentPlayerBoard = Number(player) === gameView.current_player
                      const pendingAction = canPlayHumanAction && isCurrentPlayerBoard ? getHumanActionFromPosition(idx) : null
                      const canClickCell = pendingAction !== null
                      const tone = visible ? cardTone(value) : null
                      return (
                        <button
                          type="button"
                          className={`cell ${visible ? 'visible' : 'hidden'} ${canClickCell ? 'selectable' : ''}`}
                          key={idx}
                          disabled={!canClickCell || loading}
                          style={tone ? { backgroundColor: tone.bg, color: tone.fg } : undefined}
                          onClick={() => {
                            if (pendingAction !== null) {
                              void submitHumanAction(pendingAction)
                            }
                          }}
                        >
                          <span>{visible ? value : '?'}</span>
                        </button>
                      )
                    })}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Human Turn Controls</h2>
            {canPlayHumanAction ? (
              gameView.phase === 'SETUP' ? (
                <p>Select a position on your board to reveal a setup card.</p>
              ) : (
                <div className="pile-controls">
                  <p>Choose source, preview card, then click a board position.</p>
                  <div className="actions">
                    <button
                      disabled={!hasLegalMacro('DRAW_DECK_KEEP_AND_REPLACE') || discardCommitted || loading}
                      className={selectedHumanSource === 'deck' ? 'selected' : ''}
                      onClick={() => {
                        setSelectedHumanSource('deck')
                        setDeckDecision('keep')
                        setPendingDeckDrawnValue(deckTopValue)
                        setDiscardCommitted(false)
                      }}
                    >
                      <CardBox label={`Deck (${gameView.deck_size})`} value={null} unknownLabel="?" />
                    </button>
                    <button
                      disabled={!hasLegalMacro('TAKE_DISCARD_AND_REPLACE') || discardCommitted || loading}
                      className={selectedHumanSource === 'discard' ? 'selected' : ''}
                      onClick={() => {
                        setSelectedHumanSource('discard')
                        setDeckDecision('keep')
                        setPendingDeckDrawnValue(null)
                        setDiscardCommitted(false)
                      }}
                    >
                      <CardBox
                        label="Discard"
                        value={discardCommitted && pendingDeckDrawnValue !== null ? pendingDeckDrawnValue : gameView.discard_top}
                      />
                    </button>
                    <button
                      type="button"
                      className={`draw-preview-button ${
                        selectedHumanSource === 'deck' && canDiscardDrawn ? 'can-discard' : ''
                      } ${
                        selectedHumanSource === 'deck' && deckDecision === 'discard' ? 'selected' : ''
                      }`}
                      disabled={selectedHumanSource !== 'deck' || !canDiscardDrawn || discardCommitted || loading}
                      onClick={() => {
                        if (selectedHumanSource === 'deck' && !discardCommitted) {
                          setDeckDecision('discard')
                          setDiscardCommitted(true)
                        }
                      }}
                    >
                      <CardBox
                        label="Drawn Card"
                        value={
                          selectedHumanSource === null
                            ? null
                            : selectedHumanSource === 'deck'
                              ? discardCommitted
                                ? null
                                : pendingDeckDrawnValue
                              : gameView.discard_top
                        }
                        unknownLabel="?"
                      />
                    </button>
                    {selectedHumanSource === 'deck' && (
                      <small className="draw-mode-hint">
                        Mode: {deckDecision === 'keep' ? 'Keep and replace any slot' : 'Discard and flip unknown slot'}
                      </small>
                    )}
                  </div>
                </div>
              )
            ) : (
              <p>Waiting on an AI-controlled turn.</p>
            )}
            <div className="actions">
              <button disabled={!canStepAi || loading} onClick={stepAiTurn}>
                Step AI
              </button>
              <button disabled={!gameView || gameView.terminated || loading} onClick={() => runAutoplayBurst(25)}>
                Autoplay 25
              </button>
              <button
                disabled={!gameView || gameView.terminated}
                onClick={() => {
                  setIsAutoRunning((prev) => !prev)
                }}
              >
                {isAutoRunning ? 'Pause' : 'Play'}
              </button>
              <label>
                Delay (ms)
                <input
                  type="number"
                  min={100}
                  max={5000}
                  value={autoplayDelayMs}
                  onChange={(event) => setAutoplayDelayMs(Number(event.target.value))}
                />
              </label>
              <label>
                Auto AI after human move
                <input
                  type="checkbox"
                  checked={autoRespondToAi}
                  onChange={(event) => setAutoRespondToAi(event.target.checked)}
                />
              </label>
            </div>
          </section>

          <section className="panel">
            <h2>Public Action Log</h2>
            <div className="history" ref={logHistoryRef} onScroll={handleLogScroll}>
              {actionLogLines.map((entry, idx) => (
                <div key={entry.id}>
                  {idx + 1}. {entry.text}
                </div>
              ))}
            </div>
          </section>
        </>
      ) : (
        <section className="panel">
          <p>Start a game to render cards and actions.</p>
        </section>
      )}
    </main>
  )
}

export default App
