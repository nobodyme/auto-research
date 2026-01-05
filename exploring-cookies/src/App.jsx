import { useState } from 'react'
import Navigation from './components/Navigation'
import CookieAnatomy from './components/CookieAnatomy'
import TrackingSimulator from './components/TrackingSimulator'
import SessionDemo from './components/SessionDemo'
import SecurityDemo from './components/SecurityDemo'
import BrowserPrivacy from './components/BrowserPrivacy'
import './App.css'

export default function App() {
  const [currentSection, setCurrentSection] = useState('anatomy')

  const renderSection = () => {
    switch (currentSection) {
      case 'anatomy':
        return <CookieAnatomy />
      case 'tracking':
        return <TrackingSimulator />
      case 'session':
        return <SessionDemo />
      case 'security':
        return <SecurityDemo />
      case 'privacy':
        return <BrowserPrivacy />
      default:
        return <CookieAnatomy />
    }
  }

  return (
    <div className="app-container" data-testid="app-container">
      <Navigation currentSection={currentSection} onNavigate={setCurrentSection} />
      <main className="main-content">
        {renderSection()}
        <footer className="app-footer">
          <div className="footer-content">
            <h4>Resources & Learn More</h4>
            <div className="footer-links">
              <a href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies" target="_blank" rel="noopener noreferrer">
                MDN: HTTP Cookies
              </a>
              <a href="https://web.dev/articles/samesite-cookies-explained" target="_blank" rel="noopener noreferrer">
                SameSite Cookies Explained
              </a>
              <a href="https://owasp.org/www-community/attacks/csrf" target="_blank" rel="noopener noreferrer">
                OWASP: CSRF Attacks
              </a>
              <a href="https://privacysandbox.com" target="_blank" rel="noopener noreferrer">
                Google Privacy Sandbox
              </a>
              <a href="https://webkit.org/tracking-prevention/" target="_blank" rel="noopener noreferrer">
                Safari Tracking Prevention
              </a>
            </div>
            <p className="footer-note">
              Built for educational purposes. Explore cookies responsibly. 🍪
            </p>
          </div>
        </footer>
      </main>
    </div>
  )
}
