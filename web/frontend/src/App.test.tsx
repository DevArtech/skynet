import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

describe('App', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('creates a game and renders server state', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        state: { seed: 7 },
        view: {
          terminated: false,
          phase: 'SETUP',
          current_player: 0,
          observer_player: 0,
          scores: [0, 0],
          round_scores: [0, 0],
          deck_size: 100,
          discard_top: 5,
          discard_size: 1,
          legal_actions: [{ action_id: 36, macro: 'SETUP_FLIP', position: 0, label: 'SETUP_FLIP @ 0' }],
          board_views: {
            '0': { values: Array(12).fill(-99), visible_mask: Array(12).fill(0) },
            '1': { values: Array(12).fill(-99), visible_mask: Array(12).fill(0) },
          },
          public_history: [],
        },
      }),
    } as Response)

    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start Game' }))

    await waitFor(() => expect(screen.getByText(/Phase:/)).toBeInTheDocument())

    expect(fetchMock).toHaveBeenCalled()
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/session/new')
    expect(init?.method).toBe('POST')
  })
})
