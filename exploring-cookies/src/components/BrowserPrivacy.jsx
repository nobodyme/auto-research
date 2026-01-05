import { useState } from 'react'
import './BrowserPrivacy.css'

const browsers = {
  chrome: {
    name: 'Google Chrome',
    icon: '🌐',
    color: '#4285f4',
    thirdPartyBlocked: false,
    features: [
      'Third-party cookies allowed by default',
      'SameSite=Lax is default for cookies without SameSite',
      'Privacy Sandbox APIs (deprecated Oct 2025)',
      'Incognito mode blocks third-party cookies',
    ],
    trackingAllowed: true,
    marketShare: '~60%',
    policy: 'Permissive - Users can choose in settings',
  },
  safari: {
    name: 'Safari (ITP)',
    icon: '🧭',
    color: '#0066cc',
    thirdPartyBlocked: true,
    features: [
      'All third-party cookies blocked since 2020',
      'First-party cookies capped to 7 days in cross-site contexts',
      'Link decoration tracking parameters stripped',
      'CNAME cloaking detected and blocked',
      'Machine learning identifies trackers',
    ],
    trackingAllowed: false,
    marketShare: '~20%',
    policy: 'Intelligent Tracking Prevention (ITP)',
  },
  firefox: {
    name: 'Firefox (ETP)',
    icon: '🦊',
    color: '#ff6611',
    thirdPartyBlocked: true,
    features: [
      'Total Cookie Protection - separate cookie jar per site',
      'Known trackers blocked by default',
      'Fingerprinting protection',
      'Supercookie isolation',
      'SmartBlock for compatibility',
    ],
    trackingAllowed: false,
    marketShare: '~8%',
    policy: 'Enhanced Tracking Protection (ETP)',
  },
  brave: {
    name: 'Brave',
    icon: '🦁',
    color: '#fb542b',
    thirdPartyBlocked: true,
    features: [
      'All third-party cookies blocked',
      'Aggressive fingerprinting protection',
      'Bounce tracking protection',
      'First-party cookie isolation',
      'Built-in ad and tracker blocking',
    ],
    trackingAllowed: false,
    marketShare: '~1%',
    policy: 'Shields - Maximum privacy by default',
  },
}

const timelineEvents = [
  { year: 2017, event: 'Safari introduces ITP 1.0', impact: 'First major browser to limit tracking' },
  { year: 2019, event: 'Firefox enables ETP by default', impact: 'Blocks known trackers automatically' },
  { year: 2020, event: 'Safari blocks ALL third-party cookies', impact: 'Complete tracking prevention' },
  { year: 2021, event: 'Firefox adds Total Cookie Protection', impact: 'Partitioned storage per site' },
  { year: 2024, event: 'Google abandons cookie deprecation', impact: 'Reverses 6 years of planning' },
  { year: 2025, event: 'Privacy Sandbox APIs deprecated', impact: 'Industry left without alternatives' },
]

export default function BrowserPrivacy() {
  const [selectedBrowser, setSelectedBrowser] = useState('chrome')
  const [trackingSimulated, setTrackingSimulated] = useState(false)
  const [consentGiven, setConsentGiven] = useState(null)

  const browser = browsers[selectedBrowser]

  const simulateTracking = () => {
    setTrackingSimulated(true)
  }

  const handleConsent = (accepted) => {
    setConsentGiven(accepted)
  }

  return (
    <div className="browser-privacy">
      <h2>Browser Privacy Protections</h2>
      <p className="intro">
        Different browsers take vastly different approaches to cookie privacy.
        See how your browsing experience and tracking exposure varies by browser choice.
      </p>

      <div className="browser-selector">
        {Object.entries(browsers).map(([key, b]) => (
          <button
            key={key}
            data-testid={`browser-${key}`}
            className={`browser-btn ${selectedBrowser === key ? 'selected' : ''}`}
            style={{ '--browser-color': b.color }}
            onClick={() => {
              setSelectedBrowser(key)
              setTrackingSimulated(false)
            }}
          >
            <span className="browser-icon">{b.icon}</span>
            <span className="browser-name">{b.name}</span>
            <span className="market-share">{b.marketShare}</span>
          </button>
        ))}
      </div>

      <div className="browser-details">
        <div className="browser-header" style={{ '--browser-color': browser.color }}>
          <span className="icon">{browser.icon}</span>
          <div>
            <h3>{browser.name}</h3>
            <span className="policy">{browser.policy}</span>
          </div>
          <span className={`tracking-badge ${browser.thirdPartyBlocked ? 'blocked' : 'allowed'}`}>
            {browser.thirdPartyBlocked ? '🛡️ Third-Party Blocked' : '⚠️ Third-Party Allowed'}
          </span>
        </div>

        <div className="features-list">
          <h4>Privacy Features</h4>
          <ul>
            {browser.features.map((feature, idx) => (
              <li key={idx}>{feature}</li>
            ))}
          </ul>
        </div>

        <div className="tracking-simulation">
          <h4>Tracking Simulation</h4>
          <button onClick={simulateTracking} className="simulate-btn">
            Simulate Tracking Attempt
          </button>

          {trackingSimulated && (
            <div className="tracking-result" data-testid="tracking-result">
              {browser.trackingAllowed ? (
                <div className="result allowed">
                  <span className="icon">⚠️</span>
                  <div>
                    <strong>Tracking allowed</strong>
                    <p>Third-party cookies can track you across websites</p>
                    <code>Cookie: _tracker=abc123 sent to tracker.com</code>
                  </div>
                </div>
              ) : (
                <div className="result blocked">
                  <span className="icon">🛡️</span>
                  <div>
                    <strong>Tracking blocked</strong>
                    <p>Third-party cookies rejected or partitioned</p>
                    <code>Cookie request from tracker.com: BLOCKED</code>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="partitioning-demo" data-testid="partitioning-demo">
        <h3>Cookie Partitioning Explained</h3>
        <p>
          Modern browsers like Firefox use <strong>Total Cookie Protection</strong> to prevent
          cross-site tracking while still allowing third-party cookies to function.
        </p>

        <div className="partitioning-visual">
          <div className="partition" data-testid="cookie-jar-partition">
            <div className="partition-header">
              <span>news.com</span>
            </div>
            <div className="partition-content">
              <div className="embedded-site">
                <span>tracker.com cookie:</span>
                <code>id=A1B2C3</code>
              </div>
            </div>
          </div>

          <div className="partition-separator">
            <span>Partitioned</span>
            <span>🔒</span>
          </div>

          <div className="partition" data-testid="cookie-jar-partition">
            <div className="partition-header">
              <span>shop.com</span>
            </div>
            <div className="partition-content">
              <div className="embedded-site">
                <span>tracker.com cookie:</span>
                <code>id=X7Y8Z9</code>
              </div>
            </div>
          </div>
        </div>

        <p className="partition-note">
          Same tracker, different IDs! The tracker cannot correlate these visits.
        </p>
      </div>

      <div className="comparison-section">
        <h3>Browser Comparison</h3>
        <div className="browser-comparison-table" data-testid="browser-comparison-table">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Chrome</th>
                <th>Safari</th>
                <th>Firefox</th>
                <th>Brave</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Third-party Cookies</td>
                <td className="status-allowed">Allowed</td>
                <td className="status-blocked">Blocked</td>
                <td className="status-blocked">Blocked</td>
                <td className="status-blocked">Blocked</td>
              </tr>
              <tr>
                <td>Cookie Partitioning</td>
                <td className="status-partial">CHIPS only</td>
                <td className="status-blocked">Full</td>
                <td className="status-blocked">Full</td>
                <td className="status-blocked">Full</td>
              </tr>
              <tr>
                <td>Fingerprinting Protection</td>
                <td className="status-allowed">Minimal</td>
                <td className="status-partial">Moderate</td>
                <td className="status-blocked">Strong</td>
                <td className="status-blocked">Aggressive</td>
              </tr>
              <tr>
                <td>Tracker Blocking</td>
                <td className="status-allowed">None</td>
                <td className="status-blocked">ML-based</td>
                <td className="status-blocked">List-based</td>
                <td className="status-blocked">Built-in</td>
              </tr>
              <tr>
                <td>Link Decoration Stripped</td>
                <td className="status-allowed">No</td>
                <td className="status-blocked">Yes</td>
                <td className="status-partial">Partial</td>
                <td className="status-blocked">Yes</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="consent-banner-demo" data-testid="consent-banner-demo">
        <h3>The Irony of Cookie Consent</h3>
        <p>
          GDPR requires consent for non-essential cookies. Ironically, sites use cookies
          to remember your consent choice!
        </p>

        <div className="consent-simulation">
          <div className="fake-banner">
            <p>🍪 We use cookies to improve your experience</p>
            <div className="banner-buttons">
              <button onClick={() => handleConsent(true)} className="accept-btn">
                Accept All
              </button>
              <button onClick={() => handleConsent(false)} className="reject-btn">
                Reject Non-Essential
              </button>
            </div>
          </div>

          {consentGiven !== null && (
            <div className="consent-result">
              <h4>Cookie Set to Remember Your Choice:</h4>
              <code>
                {consentGiven
                  ? 'consent_cookie=analytics:true,marketing:true; Max-Age=31536000'
                  : 'consent_cookie=analytics:false,marketing:false; Max-Age=31536000'}
              </code>
              <p className="irony-note">
                📌 You chose to {consentGiven ? 'accept' : 'reject'} cookies...
                using a cookie to remember that choice.
              </p>
            </div>
          )}
        </div>

        <div className="gpc-section">
          <h4>Global Privacy Control (GPC)</h4>
          <p>
            A browser signal that tells websites "Do Not Sell My Personal Information."
            Legally binding in California under CPRA.
          </p>
          <code>Sec-GPC: 1</code>
        </div>
      </div>

      <div className="privacy-timeline" data-testid="privacy-timeline">
        <h3>The Evolution of Cookie Privacy</h3>
        <div className="timeline">
          {timelineEvents.map((event, idx) => (
            <div key={idx} className="timeline-event">
              <div className="event-year">{event.year}</div>
              <div className="event-content">
                <strong>{event.event}</strong>
                <p>{event.impact}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="death-of-cookies">
          <h4>The "Death" of Third-Party Cookies</h4>
          <p>
            By 2026, over <strong>60% of browsers block third-party cookies</strong> by default
            (Safari, Firefox, Brave), plus 30%+ of users run ad blockers.
          </p>
          <p>
            Google Chrome (60% market share) still allows them, but the direction of the industry
            is clear: third-party tracking cookies are becoming obsolete.
          </p>
          <div className="alternatives">
            <h5>What's Replacing Third-Party Cookies?</h5>
            <ul>
              <li><strong>First-party data</strong> - Direct customer relationships</li>
              <li><strong>Server-side tracking</strong> - Facebook CAPI, Google Enhanced Conversions</li>
              <li><strong>CHIPS</strong> - Partitioned third-party cookies (limited use)</li>
              <li><strong>Unified ID 2.0</strong> - Hashed email-based identification</li>
              <li><strong>Contextual advertising</strong> - Ads based on page content, not user</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
