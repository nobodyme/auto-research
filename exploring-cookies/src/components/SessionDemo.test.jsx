import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SessionDemo from './SessionDemo'

describe('SessionDemo', () => {
  beforeEach(() => {
    render(<SessionDemo />)
  })

  it('renders the component with title', () => {
    expect(screen.getByRole('heading', { name: /Session Management/i })).toBeInTheDocument()
  })

  it('shows a login form initially', () => {
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^login$/i })).toBeInTheDocument()
  })

  it('displays cookie jar visualization', () => {
    expect(screen.getByTestId('cookie-jar')).toBeInTheDocument()
  })

  it('shows empty cookie jar before login', () => {
    const cookieJar = screen.getByTestId('cookie-jar')
    expect(within(cookieJar).queryAllByTestId('cookie-item')).toHaveLength(0)
  })

  it('creates session cookie on login', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    const cookieJar = screen.getByTestId('cookie-jar')
    expect(within(cookieJar).getAllByTestId('cookie-item').length).toBeGreaterThan(0)
  })

  it('shows session cookie details after login', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    expect(screen.getByTestId('session-cookie-details')).toBeInTheDocument()
    expect(screen.getByTestId('session-cookie-details').textContent).toContain('session')
  })

  it('displays user dashboard after login', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    expect(screen.getByText(/welcome, alice/i)).toBeInTheDocument()
  })

  it('shows HTTP request/response with Set-Cookie header', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    const httpViewer = screen.getByTestId('http-viewer')
    expect(within(httpViewer).getByText(/Set-Cookie/i)).toBeInTheDocument()
  })

  it('demonstrates "Remember Me" functionality', async () => {
    const user = userEvent.setup()

    const rememberMe = screen.getByLabelText(/remember me/i)
    await user.click(rememberMe)

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    // Should show persistent cookie with Max-Age
    const cookieDetails = screen.getByTestId('session-cookie-details')
    expect(cookieDetails.textContent).toMatch(/Max-Age|Expires/i)
  })

  it('shows logout button after login', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))

    expect(screen.getByRole('button', { name: /logout/i })).toBeInTheDocument()
  })

  it('clears session cookie on logout', async () => {
    const user = userEvent.setup()

    await user.type(screen.getByLabelText(/username/i), 'alice')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /^login$/i }))
    await user.click(screen.getByRole('button', { name: /logout/i }))

    const cookieJar = screen.getByTestId('cookie-jar')
    expect(within(cookieJar).queryAllByTestId('cookie-item')).toHaveLength(0)
  })

  it('shows shopping cart demo section', () => {
    expect(screen.getByTestId('shopping-cart-demo')).toBeInTheDocument()
  })

  it('adds items to cart using cookies', async () => {
    const user = userEvent.setup()

    const addButtons = screen.getAllByTestId('add-item-button')
    await user.click(addButtons[0])

    expect(screen.getByTestId('cart-cookie')).toBeInTheDocument()
    expect(screen.getByTestId('cart-cookie').textContent).toContain('cart')
  })

  it('persists cart across simulated page navigation', async () => {
    const user = userEvent.setup()

    const addButtons = screen.getAllByTestId('add-item-button')
    await user.click(addButtons[0])
    await user.click(screen.getByRole('button', { name: /simulate|refresh/i }))

    const cartItems = screen.getByTestId('cart-items')
    expect(within(cartItems).getAllByTestId('cart-item').length).toBeGreaterThan(0)
  })

  it('explains difference between session and persistent cookies', () => {
    expect(screen.getByText(/Session.*Persistent/i)).toBeInTheDocument()
  })
})
