import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CookieAnatomy from './CookieAnatomy'

describe('CookieAnatomy', () => {
  beforeEach(() => {
    render(<CookieAnatomy />)
  })

  it('renders the component with title', () => {
    expect(screen.getByRole('heading', { name: /Cookie Anatomy/i })).toBeInTheDocument()
  })

  it('displays cookie name input field', () => {
    expect(screen.getByLabelText(/^name$/i)).toBeInTheDocument()
  })

  it('displays cookie value input field', () => {
    expect(screen.getByLabelText(/^value$/i)).toBeInTheDocument()
  })

  it('shows all cookie attributes as toggleable options', () => {
    // Check for the text labels that exist
    expect(screen.getByText(/^Secure$/)).toBeInTheDocument()
    expect(screen.getByText(/^HttpOnly$/)).toBeInTheDocument()
    expect(screen.getByLabelText(/samesite/i)).toBeInTheDocument()
  })

  it('generates Set-Cookie header preview', () => {
    expect(screen.getByTestId('cookie-preview')).toBeInTheDocument()
  })

  it('updates preview when name changes', async () => {
    const user = userEvent.setup()
    const nameInput = screen.getByLabelText(/^name$/i)

    await user.clear(nameInput)
    await user.type(nameInput, 'testSession')

    const preview = screen.getByTestId('cookie-preview')
    expect(preview.textContent).toContain('testSession')
  })

  it('updates preview when value changes', async () => {
    const user = userEvent.setup()
    const valueInput = screen.getByLabelText(/^value$/i)

    await user.clear(valueInput)
    await user.type(valueInput, 'newValue123')

    const preview = screen.getByTestId('cookie-preview')
    expect(preview.textContent).toContain('newValue123')
  })

  it('shows Secure attribute in preview when enabled', async () => {
    const user = userEvent.setup()
    // Find the checkbox within the Secure label
    const secureLabel = screen.getByText(/^Secure$/).closest('label')
    const checkbox = secureLabel.querySelector('input[type="checkbox"]')

    await user.click(checkbox)

    const preview = screen.getByTestId('cookie-preview')
    expect(preview.textContent).toContain('Secure')
  })

  it('shows HttpOnly attribute in preview when enabled', async () => {
    const user = userEvent.setup()
    const httpOnlyLabel = screen.getByText(/^HttpOnly$/).closest('label')
    const checkbox = httpOnlyLabel.querySelector('input[type="checkbox"]')

    await user.click(checkbox)

    const preview = screen.getByTestId('cookie-preview')
    expect(preview.textContent).toContain('HttpOnly')
  })

  it('shows SameSite attribute in preview when selected', async () => {
    const user = userEvent.setup()
    const sameSiteSelect = screen.getByLabelText(/samesite/i)

    await user.selectOptions(sameSiteSelect, 'Strict')

    const preview = screen.getByTestId('cookie-preview')
    expect(preview.textContent).toContain('SameSite=Strict')
  })

  it('displays explanation for each attribute when hovered', async () => {
    const user = userEvent.setup()
    const secureLabel = screen.getByText(/^Secure$/).closest('label')

    await user.hover(secureLabel)

    expect(screen.getByRole('tooltip')).toBeInTheDocument()
  })

  it('shows security score based on attributes enabled', () => {
    expect(screen.getByTestId('security-score')).toBeInTheDocument()
  })

  it('increases security score when HttpOnly is enabled', async () => {
    const user = userEvent.setup()
    const scoreElement = screen.getByTestId('security-score')
    const initialScore = parseInt(scoreElement.textContent.match(/\d+/)?.[0] || '0')

    const httpOnlyLabel = screen.getByText(/^HttpOnly$/).closest('label')
    const checkbox = httpOnlyLabel.querySelector('input[type="checkbox"]')
    await user.click(checkbox)

    const newScoreElement = screen.getByTestId('security-score')
    const newScore = parseInt(newScoreElement.textContent.match(/\d+/)?.[0] || '0')
    expect(newScore).toBeGreaterThan(initialScore)
  })

  it('shows domain and path fields', () => {
    expect(screen.getByLabelText(/^domain$/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/^path$/i)).toBeInTheDocument()
  })

  it('shows Max-Age / Expires options', () => {
    expect(screen.getByLabelText(/max-age|expires/i)).toBeInTheDocument()
  })

  it('displays educational info about cookie types', () => {
    expect(screen.getAllByText(/Session Cookie/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Persistent Cookie/).length).toBeGreaterThan(0)
  })
})
