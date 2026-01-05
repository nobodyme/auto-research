import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SecurityDemo from './SecurityDemo'

describe('SecurityDemo', () => {
  beforeEach(() => {
    render(<SecurityDemo />)
  })

  it('renders the component with title', () => {
    expect(screen.getByRole('heading', { name: /Security Vulnerabilities/i })).toBeInTheDocument()
  })

  it('shows CSRF attack demo section', () => {
    expect(screen.getByTestId('csrf-demo')).toBeInTheDocument()
  })

  it('shows XSS attack demo section', () => {
    expect(screen.getByTestId('xss-demo')).toBeInTheDocument()
  })

  it('shows session hijacking demo section', () => {
    expect(screen.getByTestId('session-hijacking-demo')).toBeInTheDocument()
  })

  // CSRF Demo tests
  describe('CSRF Demo', () => {
    it('shows victim bank website', () => {
      const csrfDemo = screen.getByTestId('csrf-demo')
      expect(within(csrfDemo).getByTestId('bank-website')).toBeInTheDocument()
    })

    it('shows attacker website', () => {
      const csrfDemo = screen.getByTestId('csrf-demo')
      expect(within(csrfDemo).getByTestId('attacker-website')).toBeInTheDocument()
    })

    it('allows user to login to bank', async () => {
      const user = userEvent.setup()
      const csrfDemo = screen.getByTestId('csrf-demo')

      const loginBtn = within(csrfDemo).getByRole('button', { name: /login to bank/i })
      await user.click(loginBtn)

      // Check for logged in badge
      expect(within(csrfDemo).getByText(/✓ Logged In/i)).toBeInTheDocument()
    })

    it('demonstrates CSRF attack when visiting attacker site', async () => {
      const user = userEvent.setup()
      const csrfDemo = screen.getByTestId('csrf-demo')

      // Login to bank first
      await user.click(within(csrfDemo).getByRole('button', { name: /login to bank/i }))

      // Visit attacker site
      await user.click(within(csrfDemo).getByRole('button', { name: /visit malicious site/i }))

      // Should show transfer happening - multiple "attacker" words may appear
      expect(within(csrfDemo).getAllByText(/attacker/i).length).toBeGreaterThan(0)
    })

    it('shows how SameSite cookie prevents CSRF', async () => {
      const user = userEvent.setup()
      const csrfDemo = screen.getByTestId('csrf-demo')

      // Enable SameSite protection
      const checkbox = within(csrfDemo).getByRole('checkbox')
      await user.click(checkbox)

      // Login and try attack
      await user.click(within(csrfDemo).getByRole('button', { name: /login to bank/i }))
      await user.click(within(csrfDemo).getByRole('button', { name: /visit malicious site/i }))

      // Attack should be blocked - look for BLOCKED text (may appear in multiple places)
      expect(within(csrfDemo).getAllByText(/BLOCKED/i).length).toBeGreaterThan(0)
    })

    it('displays HTTP requests showing cookie behavior', async () => {
      const user = userEvent.setup()
      const csrfDemo = screen.getByTestId('csrf-demo')

      await user.click(within(csrfDemo).getByRole('button', { name: /login to bank/i }))

      expect(within(csrfDemo).getByTestId('http-log')).toBeInTheDocument()
    })
  })

  // XSS Demo tests
  describe('XSS Demo', () => {
    it('shows vulnerable comment form', () => {
      const xssDemo = screen.getByTestId('xss-demo')
      expect(within(xssDemo).getByLabelText(/comment/i)).toBeInTheDocument()
    })

    it('demonstrates cookie theft via XSS', async () => {
      const user = userEvent.setup()
      const xssDemo = screen.getByTestId('xss-demo')

      // Post malicious comment
      const commentInput = within(xssDemo).getByLabelText(/comment/i)
      await user.type(commentInput, '<script>alert(document.cookie)</script>')
      await user.click(within(xssDemo).getByRole('button', { name: /post/i }))

      // Should show stolen cookie
      expect(within(xssDemo).getByTestId('stolen-data')).toBeInTheDocument()
    })

    it('shows how HttpOnly prevents cookie theft', async () => {
      const user = userEvent.setup()
      const xssDemo = screen.getByTestId('xss-demo')

      // Enable HttpOnly protection - find checkbox in XSS demo
      const checkboxes = within(xssDemo).getAllByRole('checkbox')
      await user.click(checkboxes[0])

      // Try XSS attack
      const commentInput = within(xssDemo).getByLabelText(/comment/i)
      await user.type(commentInput, '<script>alert(document.cookie)</script>')
      await user.click(within(xssDemo).getByRole('button', { name: /post/i }))

      // Cookie should not be visible
      expect(within(xssDemo).getByText(/not accessible/i)).toBeInTheDocument()
    })

    it('visualizes what attacker can access', () => {
      const xssDemo = screen.getByTestId('xss-demo')
      expect(within(xssDemo).getByTestId('attacker-view')).toBeInTheDocument()
    })
  })

  // Session Hijacking Demo tests
  describe('Session Hijacking Demo', () => {
    it('shows network traffic visualization', () => {
      const hijackDemo = screen.getByTestId('session-hijacking-demo')
      expect(within(hijackDemo).getByTestId('network-traffic')).toBeInTheDocument()
    })

    it('demonstrates interception on HTTP', async () => {
      const user = userEvent.setup()
      const hijackDemo = screen.getByTestId('session-hijacking-demo')

      await user.click(within(hijackDemo).getByRole('button', { name: /intercept/i }))

      expect(within(hijackDemo).getByText(/captured/i)).toBeInTheDocument()
    })

    it('shows how Secure attribute prevents interception on HTTPS', async () => {
      const user = userEvent.setup()
      const hijackDemo = screen.getByTestId('session-hijacking-demo')

      // Enable Secure + HTTPS
      const checkbox = within(hijackDemo).getByRole('checkbox')
      await user.click(checkbox)
      await user.click(within(hijackDemo).getByRole('button', { name: /intercept/i }))

      // Check that encryption/cannot intercept message appears
      expect(within(hijackDemo).getAllByText(/encrypted|Cannot intercept/i).length).toBeGreaterThan(0)
    })
  })

  it('shows security best practices summary', () => {
    expect(screen.getByTestId('security-best-practices')).toBeInTheDocument()
  })

  it('displays recommended cookie configuration', () => {
    expect(screen.getByTestId('recommended-config')).toBeInTheDocument()
    // Check that there's code showing cookie configuration
    const config = screen.getByTestId('recommended-config')
    expect(config.textContent).toMatch(/Secure|HttpOnly|SameSite/)
  })
})
