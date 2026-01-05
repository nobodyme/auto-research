import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import TrackingSimulator from './TrackingSimulator'

describe('TrackingSimulator', () => {
  beforeEach(() => {
    render(<TrackingSimulator />)
  })

  it('renders the component with title', () => {
    expect(screen.getByRole('heading', { name: /Cross-Site Tracking/i })).toBeInTheDocument()
  })

  it('displays multiple simulated website tabs', () => {
    expect(screen.getByTestId('website-news')).toBeInTheDocument()
    expect(screen.getByTestId('website-shopping')).toBeInTheDocument()
    expect(screen.getByTestId('website-social')).toBeInTheDocument()
  })

  it('shows a third-party tracker visualization', () => {
    expect(screen.getByTestId('tracker-panel')).toBeInTheDocument()
  })

  it('shows user profile building as they visit sites', async () => {
    const user = userEvent.setup()

    const newsTab = screen.getByTestId('website-news')
    await user.click(newsTab)

    // Check that the tracker panel shows site visit
    const trackerPanel = screen.getByTestId('tracker-panel')
    expect(within(trackerPanel).getByText(/TechNews/i)).toBeInTheDocument()
  })

  it('accumulates interests when interacting with content', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('website-news'))

    // Click on an article to add interest
    const articles = screen.getAllByText(/Technology|Politics|Science/i)
    if (articles.length > 0) {
      await user.click(articles[0].closest('.article-card') || articles[0])
    }

    const trackerPanel = screen.getByTestId('tracker-panel')
    const interestsList = within(trackerPanel).getByTestId('interests-list')
    expect(interestsList.textContent).not.toBe('')
  })

  it('shows cookie sync visualization between ad networks', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('website-news'))

    expect(screen.getByTestId('cookie-sync-animation')).toBeInTheDocument()
  })

  it('displays the unique tracker ID assigned to user', () => {
    expect(screen.getByTestId('tracker-id')).toBeInTheDocument()
    expect(screen.getByTestId('tracker-id').textContent).toMatch(/TRK-[A-Za-z0-9]+/)
  })

  it('shows how same ID is sent across different sites', async () => {
    const user = userEvent.setup()

    const trackerId = screen.getByTestId('tracker-id').textContent

    await user.click(screen.getByTestId('website-news'))
    await user.click(screen.getByTestId('website-shopping'))

    // Same ID should be visible in requests
    const requests = screen.getAllByTestId('http-request')
    requests.forEach((request) => {
      expect(request.textContent).toContain(trackerId)
    })
  })

  it('visualizes HTTP requests with Cookie headers', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('website-news'))

    const requests = screen.getAllByTestId('http-request')
    expect(requests.length).toBeGreaterThan(0)
    expect(requests[0].textContent).toContain('Cookie:')
  })

  it('shows clear browsing data button', () => {
    expect(screen.getByRole('button', { name: /clear/i })).toBeInTheDocument()
  })

  it('resets tracking when clear button is clicked', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByTestId('website-news'))
    await user.click(screen.getByRole('button', { name: /clear/i }))

    const trackerPanel = screen.getByTestId('tracker-panel')
    const interestsList = within(trackerPanel).getByTestId('interests-list')
    // After reset, the interests list should show "Browse to reveal" text
    expect(interestsList.textContent).toContain('Browse')
  })

  it('shows ad retargeting demo when product is viewed', async () => {
    const user = userEvent.setup()

    // Visit shopping site and view a product
    await user.click(screen.getByTestId('website-shopping'))
    const product = screen.getByTestId('product-laptop')
    await user.click(product)

    // Visit news site - should show retargeted ad
    await user.click(screen.getByTestId('website-news'))

    expect(screen.getByTestId('retargeted-ad')).toBeInTheDocument()
  })

  it('displays explanation of tracking mechanism', () => {
    expect(screen.getAllByText(/Tracking Pixel/i).length).toBeGreaterThan(0)
  })
})
