import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

describe('App', () => {
  it('renders the application', () => {
    render(<App />)
    expect(screen.getByText(/Exploring Cookies/i)).toBeInTheDocument()
  })

  it('shows navigation', () => {
    render(<App />)
    expect(screen.getByRole('navigation')).toBeInTheDocument()
  })

  it('defaults to showing introduction/anatomy section', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: /Cookie Anatomy/i })).toBeInTheDocument()
  })

  it('can navigate to tracking section', async () => {
    const user = userEvent.setup()
    render(<App />)

    const nav = screen.getByRole('navigation')
    await user.click(within(nav).getByText(/^Tracking$/i))

    expect(screen.getByRole('heading', { name: /Cross-Site Tracking/i })).toBeInTheDocument()
  })

  it('can navigate to session section', async () => {
    const user = userEvent.setup()
    render(<App />)

    const nav = screen.getByRole('navigation')
    await user.click(within(nav).getByText(/^Session$/i))

    expect(screen.getByRole('heading', { name: /Session Management/i })).toBeInTheDocument()
  })

  it('can navigate to security section', async () => {
    const user = userEvent.setup()
    render(<App />)

    const nav = screen.getByRole('navigation')
    await user.click(within(nav).getByText(/^Security$/i))

    expect(screen.getByRole('heading', { name: /Security Vulnerabilities/i })).toBeInTheDocument()
  })

  it('can navigate to privacy section', async () => {
    const user = userEvent.setup()
    render(<App />)

    const nav = screen.getByRole('navigation')
    await user.click(within(nav).getByText(/^Privacy$/i))

    expect(screen.getByRole('heading', { name: /Browser Privacy/i })).toBeInTheDocument()
  })

  it('shows footer with additional resources', () => {
    render(<App />)
    expect(screen.getByText(/Resources.*Learn More/i)).toBeInTheDocument()
  })

  it('has responsive design classes', () => {
    render(<App />)
    const container = screen.getByTestId('app-container')
    expect(container).toHaveClass('app-container')
  })
})
