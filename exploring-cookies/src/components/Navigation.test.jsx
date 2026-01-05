import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Navigation from './Navigation'

describe('Navigation', () => {
  const mockOnNavigate = vi.fn()
  const defaultProps = {
    currentSection: 'anatomy',
    onNavigate: mockOnNavigate,
  }

  beforeEach(() => {
    mockOnNavigate.mockClear()
    render(<Navigation {...defaultProps} />)
  })

  it('renders navigation component', () => {
    expect(screen.getByRole('navigation')).toBeInTheDocument()
  })

  it('shows all section links', () => {
    expect(screen.getByText(/^Anatomy$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Tracking$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Session$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Security$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Privacy$/i)).toBeInTheDocument()
  })

  it('highlights current section', () => {
    const anatomyButton = screen.getByText(/^Anatomy$/i).closest('button')
    expect(anatomyButton).toHaveClass('active')
  })

  it('calls onNavigate when clicking a section', async () => {
    const user = userEvent.setup()

    await user.click(screen.getByText(/^Tracking$/i))

    expect(mockOnNavigate).toHaveBeenCalledWith('tracking')
  })

  it('shows site title', () => {
    expect(screen.getByText(/Exploring Cookies/i)).toBeInTheDocument()
  })

  it('shows progress indicator', () => {
    expect(screen.getByTestId('progress-indicator')).toBeInTheDocument()
  })
})
