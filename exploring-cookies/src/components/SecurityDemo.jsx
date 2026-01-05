import { useState } from 'react'
import './SecurityDemo.css'

export default function SecurityDemo() {
  // CSRF Demo State
  const [bankLoggedIn, setBankLoggedIn] = useState(false)
  const [bankBalance, setBankBalance] = useState(10000)
  const [sameSiteEnabled, setSameSiteEnabled] = useState(false)
  const [csrfLog, setCsrfLog] = useState([])
  const [csrfAttackResult, setCsrfAttackResult] = useState(null)

  // XSS Demo State
  const [httpOnlyEnabled, setHttpOnlyEnabled] = useState(false)
  const [comments, setComments] = useState([])
  const [commentInput, setCommentInput] = useState('')
  const [stolenData, setStolenData] = useState(null)
  const [xssSessionCookie] = useState('user_session=abc123xyz789')

  // Session Hijacking State
  const [secureHttpsEnabled, setSecureHttpsEnabled] = useState(false)
  const [interceptResult, setInterceptResult] = useState(null)
  const [networkPackets, setNetworkPackets] = useState([])

  // CSRF Functions
  const loginToBank = () => {
    setBankLoggedIn(true)
    setCsrfLog((prev) => [
      ...prev,
      {
        type: 'info',
        message: 'Logged into SecureBank',
        cookie: sameSiteEnabled
          ? 'session=abc123; SameSite=Strict; Secure; HttpOnly'
          : 'session=abc123; Secure; HttpOnly',
      },
    ])
    setCsrfAttackResult(null)
  }

  const visitMaliciousSite = () => {
    if (!bankLoggedIn) {
      setCsrfAttackResult({ success: false, message: 'User not logged in - no cookies to exploit' })
      return
    }

    if (sameSiteEnabled) {
      setCsrfLog((prev) => [
        ...prev,
        { type: 'blocked', message: 'Cross-site request blocked - SameSite cookie not sent' },
      ])
      setCsrfAttackResult({
        success: false,
        message: 'Attack BLOCKED! SameSite=Strict prevented the cookie from being sent.',
      })
    } else {
      setBankBalance((prev) => prev - 5000)
      setCsrfLog((prev) => [
        ...prev,
        { type: 'attack', message: 'Attacker forged POST to /transfer' },
        { type: 'danger', message: 'Cookie automatically sent! $5000 transferred to attacker!' },
      ])
      setCsrfAttackResult({
        success: true,
        message: 'Attack SUCCESS! $5000 was transferred to attacker account!',
      })
    }
  }

  const resetCsrf = () => {
    setBankLoggedIn(false)
    setBankBalance(10000)
    setCsrfLog([])
    setCsrfAttackResult(null)
  }

  // XSS Functions
  const postComment = () => {
    if (!commentInput.trim()) return

    const hasScript = commentInput.includes('<script')

    if (hasScript) {
      setComments((prev) => [...prev, { text: commentInput, isXss: true }])

      if (httpOnlyEnabled) {
        setStolenData({
          success: false,
          message: 'document.cookie returned empty! HttpOnly cookies are not accessible.',
        })
      } else {
        setStolenData({
          success: true,
          message: `Stolen cookie: ${xssSessionCookie}`,
          cookie: xssSessionCookie,
        })
      }
    } else {
      setComments((prev) => [...prev, { text: commentInput, isXss: false }])
    }

    setCommentInput('')
  }

  const resetXss = () => {
    setComments([])
    setCommentInput('')
    setStolenData(null)
  }

  // Session Hijacking Functions
  const attemptIntercept = () => {
    const packet = {
      timestamp: new Date().toLocaleTimeString(),
      protocol: secureHttpsEnabled ? 'HTTPS' : 'HTTP',
      from: 'User Browser',
      to: 'Bank Server',
    }

    if (secureHttpsEnabled) {
      setNetworkPackets((prev) => [
        ...prev,
        { ...packet, data: '[ENCRYPTED TLS DATA]', intercepted: false },
      ])
      setInterceptResult({
        success: false,
        message: 'Cannot intercept - data is encrypted with TLS. Cookie marked Secure only sent over HTTPS.',
      })
    } else {
      setNetworkPackets((prev) => [
        ...prev,
        { ...packet, data: 'Cookie: session=abc123xyz', intercepted: true },
      ])
      setInterceptResult({
        success: true,
        message: 'Session CAPTURED! Attacker now has the session cookie and can impersonate the user.',
        cookie: 'session=abc123xyz',
      })
    }
  }

  const resetHijacking = () => {
    setNetworkPackets([])
    setInterceptResult(null)
  }

  return (
    <div className="security-demo">
      <h2>Security Vulnerabilities</h2>
      <p className="intro">
        Explore how cookies can be exploited and how security attributes protect against attacks.
        These interactive demos show real attack vectors.
      </p>

      {/* CSRF Demo */}
      <div className="csrf-demo" data-testid="csrf-demo">
        <h3>🎯 Cross-Site Request Forgery (CSRF)</h3>
        <p className="attack-description">
          CSRF tricks your browser into making requests to sites where you're logged in.
          Your cookies are automatically sent, so the malicious request appears legitimate.
        </p>

        <div className="demo-split">
          <div className="demo-panel bank-website" data-testid="bank-website">
            <div className="panel-header">
              <span className="site-icon">🏦</span>
              <span>SecureBank.com</span>
            </div>
            <div className="panel-content">
              {!bankLoggedIn ? (
                <button onClick={loginToBank} className="demo-button primary">
                  Login to Bank
                </button>
              ) : (
                <div className="bank-dashboard">
                  <div className="balance">
                    <span className="label">Balance:</span>
                    <span className="amount">${bankBalance.toLocaleString()}</span>
                  </div>
                  <span className="logged-badge">✓ Logged In</span>
                </div>
              )}
            </div>
          </div>

          <div className="demo-panel attacker-website" data-testid="attacker-website">
            <div className="panel-header evil">
              <span className="site-icon">😈</span>
              <span>evil-site.com</span>
            </div>
            <div className="panel-content">
              <p className="evil-text">Win a FREE iPhone!</p>
              <pre className="attack-code">
                {`<form action="bank.com/transfer" method="POST">
  <input type="hidden" name="to" value="attacker">
  <input type="hidden" name="amount" value="5000">
</form>
<script>document.forms[0].submit()</script>`}
              </pre>
              <button onClick={visitMaliciousSite} className="demo-button danger">
                Visit Malicious Site
              </button>
            </div>
          </div>
        </div>

        <div className="protection-toggle">
          <label>
            <input
              type="checkbox"
              checked={sameSiteEnabled}
              onChange={(e) => setSameSiteEnabled(e.target.checked)}
            />
            Enable SameSite=Strict Protection
          </label>
          {sameSiteEnabled && (
            <span className="protection-badge">🛡️ Protected</span>
          )}
        </div>

        {csrfAttackResult && (
          <div className={`attack-result ${csrfAttackResult.success ? 'danger' : 'success'}`}>
            {csrfAttackResult.success ? '⚠️' : '✅'} {csrfAttackResult.message}
          </div>
        )}

        <div className="http-log" data-testid="http-log">
          {csrfLog.map((entry, idx) => (
            <div key={idx} className={`log-entry ${entry.type}`}>
              {entry.message}
              {entry.cookie && <code>{entry.cookie}</code>}
            </div>
          ))}
        </div>

        <button onClick={resetCsrf} className="reset-button">Reset Demo</button>
      </div>

      {/* XSS Demo */}
      <div className="xss-demo" data-testid="xss-demo">
        <h3>💉 Cross-Site Scripting (XSS)</h3>
        <p className="attack-description">
          XSS injects malicious JavaScript into a website. If successful, it can steal cookies
          by reading document.cookie - unless HttpOnly is set.
        </p>

        <div className="demo-panel comment-section">
          <div className="panel-header">
            <span className="site-icon">💬</span>
            <span>Blog Comments</span>
          </div>
          <div className="panel-content">
            <div className="comments-list">
              {comments.length === 0 ? (
                <p className="empty">No comments yet. Try posting one!</p>
              ) : (
                comments.map((comment, idx) => (
                  <div key={idx} className={`comment ${comment.isXss ? 'malicious' : ''}`}>
                    {comment.isXss ? (
                      <span className="xss-indicator">⚠️ XSS Payload Executed</span>
                    ) : null}
                    <p>{comment.text}</p>
                  </div>
                ))
              )}
            </div>

            <div className="comment-form">
              <label htmlFor="comment-input">Add Comment:</label>
              <textarea
                id="comment-input"
                value={commentInput}
                onChange={(e) => setCommentInput(e.target.value)}
                placeholder="Try: <script>alert(document.cookie)</script>"
              />
              <button onClick={postComment} className="demo-button primary">
                Post Comment
              </button>
            </div>
          </div>
        </div>

        <div className="protection-toggle">
          <label>
            <input
              type="checkbox"
              checked={httpOnlyEnabled}
              onChange={(e) => setHttpOnlyEnabled(e.target.checked)}
            />
            Enable HttpOnly Cookie Flag
          </label>
          {httpOnlyEnabled && (
            <span className="protection-badge">🛡️ Protected</span>
          )}
        </div>

        <div className="attacker-view" data-testid="attacker-view">
          <h4>👁️ Attacker's View</h4>
          <div className="stolen-data" data-testid="stolen-data">
            {stolenData ? (
              <div className={stolenData.success ? 'stolen' : 'blocked'}>
                {stolenData.success ? '🔓' : '🔒'} {stolenData.message}
              </div>
            ) : (
              <p className={httpOnlyEnabled ? 'protected' : 'vulnerable'}>
                {httpOnlyEnabled
                  ? 'Cookie not accessible - HttpOnly enabled'
                  : 'Waiting to capture document.cookie...'}
              </p>
            )}
          </div>
        </div>

        <button onClick={resetXss} className="reset-button">Reset Demo</button>
      </div>

      {/* Session Hijacking Demo */}
      <div className="session-hijacking-demo" data-testid="session-hijacking-demo">
        <h3>🕵️ Session Hijacking (Man-in-the-Middle)</h3>
        <p className="attack-description">
          On unencrypted HTTP connections, attackers can intercept network traffic and steal
          session cookies. The Secure flag ensures cookies only travel over HTTPS.
        </p>

        <div className="network-visualization" data-testid="network-traffic">
          <div className="network-node user">
            <span className="node-icon">👤</span>
            <span>User</span>
          </div>
          <div className="network-line">
            <span className="protocol-label">
              {secureHttpsEnabled ? '🔒 HTTPS' : '⚠️ HTTP'}
            </span>
            <div className="attacker-node">
              <span className="node-icon">🦹</span>
              <span>Attacker</span>
            </div>
          </div>
          <div className="network-node server">
            <span className="node-icon">🖥️</span>
            <span>Server</span>
          </div>
        </div>

        <div className="packets-log">
          {networkPackets.map((packet, idx) => (
            <div key={idx} className={`packet ${packet.intercepted ? 'intercepted' : 'encrypted'}`}>
              <span className="packet-protocol">{packet.protocol}</span>
              <span className="packet-data">{packet.data}</span>
            </div>
          ))}
        </div>

        <div className="protection-toggle">
          <label>
            <input
              type="checkbox"
              checked={secureHttpsEnabled}
              onChange={(e) => setSecureHttpsEnabled(e.target.checked)}
            />
            Enable Secure Flag + HTTPS
          </label>
          {secureHttpsEnabled && (
            <span className="protection-badge">🛡️ Protected</span>
          )}
        </div>

        <button onClick={attemptIntercept} className="demo-button danger">
          Attempt to Intercept Traffic
        </button>

        {interceptResult && (
          <div className={`attack-result ${interceptResult.success ? 'danger' : 'success'}`}>
            {interceptResult.success ? '⚠️' : '✅'} {interceptResult.message}
          </div>
        )}

        <button onClick={resetHijacking} className="reset-button">Reset Demo</button>
      </div>

      {/* Best Practices */}
      <div className="security-best-practices" data-testid="security-best-practices">
        <h3>🔐 Security Best Practices</h3>

        <div className="recommended-config" data-testid="recommended-config">
          <h4>Recommended Cookie Configuration</h4>
          <pre>
            <code>
{`Set-Cookie: __Host-session=<token>;
    Path=/;
    Secure;
    HttpOnly;
    SameSite=Strict;
    Max-Age=3600`}
            </code>
          </pre>

          <div className="config-breakdown">
            <div className="config-item">
              <strong>__Host- prefix</strong>
              <p>Prevents domain override attacks</p>
            </div>
            <div className="config-item">
              <strong>Secure</strong>
              <p>Only sent over HTTPS - prevents MITM</p>
            </div>
            <div className="config-item">
              <strong>HttpOnly</strong>
              <p>No JavaScript access - prevents XSS theft</p>
            </div>
            <div className="config-item">
              <strong>SameSite=Strict</strong>
              <p>Not sent on cross-site requests - prevents CSRF</p>
            </div>
            <div className="config-item">
              <strong>Short Max-Age</strong>
              <p>Limits exposure window if compromised</p>
            </div>
          </div>
        </div>

        <div className="attack-matrix">
          <h4>Protection Matrix</h4>
          <table>
            <thead>
              <tr>
                <th>Attack</th>
                <th>Defense</th>
                <th>Cookie Attribute</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>CSRF</td>
                <td>Block cross-site requests</td>
                <td><code>SameSite=Strict</code></td>
              </tr>
              <tr>
                <td>XSS Cookie Theft</td>
                <td>Block JavaScript access</td>
                <td><code>HttpOnly</code></td>
              </tr>
              <tr>
                <td>Man-in-the-Middle</td>
                <td>Encrypt with TLS</td>
                <td><code>Secure</code></td>
              </tr>
              <tr>
                <td>Subdomain Attack</td>
                <td>Lock to exact host</td>
                <td><code>__Host-</code> prefix</td>
              </tr>
              <tr>
                <td>Session Fixation</td>
                <td>Regenerate on login</td>
                <td>Server-side logic</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
