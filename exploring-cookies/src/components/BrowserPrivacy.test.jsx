import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import BrowserPrivacy from './BrowserPrivacy'

describe('BrowserPrivacy', () => {
  beforeEach(() => {
    render(<BrowserPrivacy />)
  })

  it('renders the component with title', () => {
    expect(screen.getByRole('heading', { name: /Browser Privacy/i })).toBeInTheDocument()
  })

  it('shows different browser icons', () => {
    expect(screen.getByTestId('browser-chrome')).toBeInTheDocument()
    expect(screen.getByTestId('browser-safari')).toBeInTheDocument()
    expect(screen.getByTestId('browser-firefox')).toBeInTheDocument()
    expect(screen.getByTestId('browser-brave')).toBeInTheDocument()
  })

  it('allows selecting different browsers', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-safari'))

    expect(screen.getByTestId('browser-safari')).toHaveClass('selected')
  })

  it('shows browser-specific cookie policies', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-safari'))

    // Safari shows ITP policy - there may be multiple mentions
    expect(screen.getAllByText(/ITP|Intelligent Tracking Prevention/i).length).toBeGreaterThan(0)
  })

  it('simulates tracking attempt with different browsers', async () => {
    const user = userEvent.setup()

    // Select Safari (strict privacy)
    await user.click(screen.getByTestId('browser-safari'))
    await user.click(screen.getByRole('button', { name: /simulate tracking/i }))

    expect(screen.getByTestId('tracking-result')).toBeInTheDocument()
    expect(screen.getByTestId('tracking-result').textContent.toLowerCase()).toContain('blocked')
  })

  it('shows Chrome allows more tracking', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-chrome'))
    await user.click(screen.getByRole('button', { name: /simulate tracking/i }))

    expect(screen.getByTestId('tracking-result').textContent.toLowerCase()).toContain('allowed')
  })

  it('displays cookie partitioning visualization', () => {
    expect(screen.getByTestId('partitioning-demo')).toBeInTheDocument()
  })

  it('shows how Firefox partitions cookies', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-firefox'))

    expect(screen.getAllByText(/Total Cookie Protection/i).length).toBeGreaterThan(0)
  })

  it('visualizes separate cookie jars per site in Firefox', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-firefox'))

    const cookieJars = screen.getAllByTestId('cookie-jar-partition')
    expect(cookieJars.length).toBeGreaterThan(1)
  })

  it('shows Safari ITP restrictions', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('browser-safari'))

    // Safari features mention 7-day cap
    expect(screen.getByText(/7.day|capped/i)).toBeInTheDocument()
  })

  it('displays comparison table of browser features', () => {
    expect(screen.getByTestId('browser-comparison-table')).toBeInTheDocument()
  })

  it('shows third-party cookie blocking status for each browser', () => {
    const table = screen.getByTestId('browser-comparison-table')

    expect(within(table).getByText(/Third-party/i)).toBeInTheDocument()
  })

  it('displays GDPR/consent banner simulation', () => {
    expect(screen.getByTestId('consent-banner-demo')).toBeInTheDocument()
  })

  it('shows irony of cookie banner using cookies', async () => {
    const user = userEvent.setup()

    const consentDemo = screen.getByTestId('consent-banner-demo')
    const buttons = within(consentDemo).getAllByRole('button')
    await user.click(buttons[0]) // Click Accept or Reject

    // Should mention consent cookie - there may be multiple mentions
    expect(within(consentDemo).getAllByText(/consent/i).length).toBeGreaterThan(0)
  })

  it('displays Global Privacy Control (GPC) info', () => {
    expect(screen.getByText(/Global Privacy Control/i)).toBeInTheDocument()
  })

  it('shows privacy timeline/evolution', () => {
    expect(screen.getByTestId('privacy-timeline')).toBeInTheDocument()
  })

  it('explains the death of third-party cookies', () => {
    // Check for text about third-party cookie blocking
    expect(screen.getByText(/60%.*block/i)).toBeInTheDocument()
  })
})
