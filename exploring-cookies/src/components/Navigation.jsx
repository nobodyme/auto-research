import './Navigation.css'

const sections = [
  { id: 'anatomy', label: 'Anatomy', icon: '🔬', description: 'Build & understand cookies' },
  { id: 'tracking', label: 'Tracking', icon: '🕵️', description: 'Cross-site tracking simulation' },
  { id: 'session', label: 'Session', icon: '🔐', description: 'Auth & shopping carts' },
  { id: 'security', label: 'Security', icon: '🛡️', description: 'CSRF, XSS & hijacking' },
  { id: 'privacy', label: 'Privacy', icon: '🏛️', description: 'Browser protections' },
]

export default function Navigation({ currentSection, onNavigate }) {
  const currentIndex = sections.findIndex((s) => s.id === currentSection)
  const progress = ((currentIndex + 1) / sections.length) * 100

  return (
    <nav className="navigation" role="navigation">
      <div className="nav-header">
        <h1 className="logo">
          <span className="cookie-emoji">🍪</span>
          Exploring Cookies
        </h1>
        <p className="tagline">An interactive guide to HTTP cookies</p>
      </div>

      <div className="progress-indicator" data-testid="progress-indicator">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-text">{currentIndex + 1} of {sections.length}</span>
      </div>

      <ul className="nav-links">
        {sections.map((section) => (
          <li key={section.id}>
            <button
              className={`nav-link ${currentSection === section.id ? 'active' : ''}`}
              onClick={() => onNavigate(section.id)}
            >
              <span className="nav-icon">{section.icon}</span>
              <div className="nav-text">
                <span className="nav-label">{section.label}</span>
                <span className="nav-description">{section.description}</span>
              </div>
            </button>
          </li>
        ))}
      </ul>

      <div className="nav-footer">
        <a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies" target="_blank" rel="noopener noreferrer">
          MDN Cookies Guide
        </a>
        <a href="https://tools.ietf.org/html/rfc6265" target="_blank" rel="noopener noreferrer">
          RFC 6265
        </a>
      </div>
    </nav>
  )
}
