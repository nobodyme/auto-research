import { useState } from 'react'
import './CookieAnatomy.css'

const attributeExplanations = {
  name: 'The identifier for the cookie. Use descriptive names like "sessionId" or "userPreferences".',
  value: 'The data stored in the cookie. Keep it short (max 4KB total with attributes).',
  domain: 'Which domain can access this cookie. Omit to restrict to current host only. Setting it explicitly includes all subdomains.',
  path: 'URL path where cookie is sent. "/" means entire site. NOT a security boundary!',
  secure: 'Cookie only sent over HTTPS connections. Essential for any sensitive data.',
  httpOnly: 'Cookie cannot be accessed by JavaScript (document.cookie). Prevents XSS from stealing cookies.',
  sameSite: 'Controls cross-site request behavior. Strict blocks all cross-site. Lax allows safe navigation. None allows all (requires Secure).',
  maxAge: 'How long until cookie expires (in seconds). Without this, it becomes a session cookie deleted when browser closes.',
}

const sameSiteExplanations = {
  '': 'Not set - defaults to Lax in modern browsers',
  'Strict': 'Never sent on cross-site requests. Most secure but can break login flows from external links.',
  'Lax': 'Sent on top-level navigation (clicking links) but not on embedded requests (images, iframes). Good balance.',
  'None': 'Sent on all requests including cross-site. Requires Secure flag. Used for third-party integrations.',
}

export default function CookieAnatomy() {
  const [cookie, setCookie] = useState({
    name: 'sessionId',
    value: 'abc123xyz',
    domain: '',
    path: '/',
    secure: false,
    httpOnly: false,
    sameSite: '',
    maxAge: '',
  })

  const [hoveredAttr, setHoveredAttr] = useState(null)
  const [useHostPrefix, setUseHostPrefix] = useState(false)
  const [useSecurePrefix, setUseSecurePrefix] = useState(false)

  const calculateSecurityScore = () => {
    let score = 0
    if (cookie.secure) score += 25
    if (cookie.httpOnly) score += 25
    if (cookie.sameSite === 'Strict') score += 30
    else if (cookie.sameSite === 'Lax') score += 20
    if (useHostPrefix) score += 20
    else if (useSecurePrefix) score += 10
    if (cookie.maxAge && parseInt(cookie.maxAge) < 3600) score += 10
    return Math.min(score, 100)
  }

  const generateCookieString = () => {
    let prefix = ''
    if (useHostPrefix) prefix = '__Host-'
    else if (useSecurePrefix) prefix = '__Secure-'

    let parts = [`${prefix}${cookie.name}=${cookie.value}`]

    if (cookie.domain && !useHostPrefix) parts.push(`Domain=${cookie.domain}`)
    if (cookie.path) parts.push(`Path=${cookie.path}`)
    if (cookie.maxAge) parts.push(`Max-Age=${cookie.maxAge}`)
    if (cookie.secure || useHostPrefix || useSecurePrefix) parts.push('Secure')
    if (cookie.httpOnly) parts.push('HttpOnly')
    if (cookie.sameSite) parts.push(`SameSite=${cookie.sameSite}`)

    return `Set-Cookie: ${parts.join('; ')}`
  }

  const securityScore = calculateSecurityScore()

  const getCookieType = () => {
    if (cookie.maxAge) {
      return { type: 'Persistent Cookie', desc: 'Survives browser restart' }
    }
    return { type: 'Session Cookie', desc: 'Deleted when browser closes' }
  }

  const cookieType = getCookieType()

  return (
    <div className="cookie-anatomy">
      <h2>Cookie Anatomy</h2>
      <p className="intro">
        Build a cookie interactively and understand how each attribute affects security and behavior.
      </p>

      <div className="anatomy-grid">
        <div className="builder-section">
          <h3>Cookie Builder</h3>

          <div className="form-group">
            <label htmlFor="cookie-name">Name</label>
            <input
              id="cookie-name"
              type="text"
              value={cookie.name}
              onChange={(e) => setCookie({ ...cookie, name: e.target.value })}
              onMouseEnter={() => setHoveredAttr('name')}
              onMouseLeave={() => setHoveredAttr(null)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="cookie-value">Value</label>
            <input
              id="cookie-value"
              type="text"
              value={cookie.value}
              onChange={(e) => setCookie({ ...cookie, value: e.target.value })}
              onMouseEnter={() => setHoveredAttr('value')}
              onMouseLeave={() => setHoveredAttr(null)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="cookie-domain">Domain</label>
            <input
              id="cookie-domain"
              type="text"
              value={cookie.domain}
              placeholder="(current host only)"
              onChange={(e) => setCookie({ ...cookie, domain: e.target.value })}
              onMouseEnter={() => setHoveredAttr('domain')}
              onMouseLeave={() => setHoveredAttr(null)}
              disabled={useHostPrefix}
            />
            {useHostPrefix && <small>__Host- prefix requires no Domain</small>}
          </div>

          <div className="form-group">
            <label htmlFor="cookie-path">Path</label>
            <input
              id="cookie-path"
              type="text"
              value={cookie.path}
              onChange={(e) => setCookie({ ...cookie, path: e.target.value })}
              onMouseEnter={() => setHoveredAttr('path')}
              onMouseLeave={() => setHoveredAttr(null)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="cookie-maxage">Max-Age / Expires</label>
            <input
              id="cookie-maxage"
              type="text"
              value={cookie.maxAge}
              placeholder="(session cookie if empty)"
              onChange={(e) => setCookie({ ...cookie, maxAge: e.target.value })}
              onMouseEnter={() => setHoveredAttr('maxAge')}
              onMouseLeave={() => setHoveredAttr(null)}
            />
            <div className="quick-options">
              <button type="button" onClick={() => setCookie({ ...cookie, maxAge: '3600' })}>1 hour</button>
              <button type="button" onClick={() => setCookie({ ...cookie, maxAge: '86400' })}>1 day</button>
              <button type="button" onClick={() => setCookie({ ...cookie, maxAge: '604800' })}>1 week</button>
              <button type="button" onClick={() => setCookie({ ...cookie, maxAge: '' })}>Session</button>
            </div>
          </div>

          <div className="security-options">
            <h4>Security Attributes</h4>

            <label
              className="checkbox-label"
              onMouseEnter={() => setHoveredAttr('secure')}
              onMouseLeave={() => setHoveredAttr(null)}
            >
              <input
                type="checkbox"
                checked={cookie.secure || useHostPrefix || useSecurePrefix}
                onChange={(e) => setCookie({ ...cookie, secure: e.target.checked })}
                disabled={useHostPrefix || useSecurePrefix}
              />
              Secure
              <span className="badge https">HTTPS only</span>
            </label>

            <label
              className="checkbox-label"
              onMouseEnter={() => setHoveredAttr('httpOnly')}
              onMouseLeave={() => setHoveredAttr(null)}
            >
              <input
                type="checkbox"
                checked={cookie.httpOnly}
                onChange={(e) => setCookie({ ...cookie, httpOnly: e.target.checked })}
              />
              HttpOnly
              <span className="badge xss">Prevents XSS</span>
            </label>

            <div
              className="form-group samesite-group"
              onMouseEnter={() => setHoveredAttr('sameSite')}
              onMouseLeave={() => setHoveredAttr(null)}
            >
              <label htmlFor="cookie-samesite">SameSite</label>
              <select
                id="cookie-samesite"
                value={cookie.sameSite}
                onChange={(e) => setCookie({ ...cookie, sameSite: e.target.value })}
              >
                <option value="">Not set (defaults to Lax)</option>
                <option value="Strict">Strict</option>
                <option value="Lax">Lax</option>
                <option value="None">None</option>
              </select>
              {cookie.sameSite && (
                <small className="samesite-note">{sameSiteExplanations[cookie.sameSite]}</small>
              )}
            </div>
          </div>

          <div className="prefix-options">
            <h4>Cookie Prefixes</h4>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useHostPrefix}
                onChange={(e) => {
                  setUseHostPrefix(e.target.checked)
                  if (e.target.checked) {
                    setUseSecurePrefix(false)
                    setCookie({ ...cookie, domain: '', path: '/' })
                  }
                }}
              />
              __Host- prefix
              <span className="badge strong">Strongest</span>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useSecurePrefix}
                onChange={(e) => {
                  setUseSecurePrefix(e.target.checked)
                  if (e.target.checked) setUseHostPrefix(false)
                }}
                disabled={useHostPrefix}
              />
              __Secure- prefix
            </label>
          </div>
        </div>

        <div className="preview-section">
          <h3>Generated Header</h3>
          <div className="cookie-preview" data-testid="cookie-preview">
            <code>{generateCookieString()}</code>
          </div>

          <div className="cookie-type-info">
            <strong>{cookieType.type}</strong> - {cookieType.desc}
            <p className="type-note">
              {cookie.maxAge
                ? `Expires in ${parseInt(cookie.maxAge)} seconds (${Math.round(parseInt(cookie.maxAge) / 3600)} hours)`
                : 'This cookie will be deleted when the browser is closed'
              }
            </p>
          </div>

          <div className="security-score" data-testid="security-score">
            <h4>Security Score</h4>
            <div className="score-bar">
              <div
                className="score-fill"
                style={{
                  width: `${securityScore}%`,
                  backgroundColor: securityScore > 70 ? '#22c55e' : securityScore > 40 ? '#f59e0b' : '#ef4444'
                }}
              />
            </div>
            <span className="score-number">{securityScore}</span>
            <p className="score-tip">
              {securityScore < 40 && '⚠️ Add Secure, HttpOnly, and SameSite for better security'}
              {securityScore >= 40 && securityScore < 70 && '🔸 Good start! Consider stricter SameSite or __Host- prefix'}
              {securityScore >= 70 && '✅ Great security configuration!'}
            </p>
          </div>

          {hoveredAttr && (
            <div className="tooltip" role="tooltip">
              <strong>{hoveredAttr}</strong>
              <p>{attributeExplanations[hoveredAttr]}</p>
            </div>
          )}

          <div className="educational-section">
            <h4>Session Cookie vs Persistent Cookie</h4>
            <div className="comparison">
              <div className="comparison-item">
                <strong>Session Cookie</strong>
                <ul>
                  <li>No Max-Age or Expires</li>
                  <li>Deleted when browser closes</li>
                  <li>Used for: temporary auth, shopping carts</li>
                </ul>
              </div>
              <div className="comparison-item">
                <strong>Persistent Cookie</strong>
                <ul>
                  <li>Has Max-Age or Expires</li>
                  <li>Survives browser restart</li>
                  <li>Used for: "remember me", preferences</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="real-world-example">
            <h4>Real-World Examples</h4>
            <button onClick={() => {
              setCookie({
                name: 'session',
                value: 'eyJhbGciOiJIUzI1NiIs...',
                domain: '',
                path: '/',
                secure: true,
                httpOnly: true,
                sameSite: 'Strict',
                maxAge: '3600',
              })
              setUseHostPrefix(true)
              setUseSecurePrefix(false)
            }}>
              Secure Session Token
            </button>
            <button onClick={() => {
              setCookie({
                name: 'theme',
                value: 'dark',
                domain: '',
                path: '/',
                secure: false,
                httpOnly: false,
                sameSite: 'Lax',
                maxAge: '31536000',
              })
              setUseHostPrefix(false)
              setUseSecurePrefix(false)
            }}>
              User Preference
            </button>
            <button onClick={() => {
              setCookie({
                name: '_ga',
                value: 'GA1.1.1234567890.1640000000',
                domain: '.example.com',
                path: '/',
                secure: false,
                httpOnly: false,
                sameSite: '',
                maxAge: '63072000',
              })
              setUseHostPrefix(false)
              setUseSecurePrefix(false)
            }}>
              Analytics Cookie
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
