import { useState } from 'react'
import './SessionDemo.css'

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 15)
}

const products = [
  { id: 1, name: 'Mechanical Keyboard', price: 149, emoji: '⌨️' },
  { id: 2, name: 'Ergonomic Mouse', price: 79, emoji: '🖱️' },
  { id: 3, name: 'USB-C Hub', price: 59, emoji: '🔌' },
  { id: 4, name: 'Monitor Stand', price: 45, emoji: '🖥️' },
]

export default function SessionDemo() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [rememberMe, setRememberMe] = useState(false)
  const [currentUser, setCurrentUser] = useState(null)
  const [sessionCookie, setSessionCookie] = useState(null)
  const [cookies, setCookies] = useState([])
  const [httpLog, setHttpLog] = useState([])
  const [cart, setCart] = useState([])
  const [cartCookie, setCartCookie] = useState(null)
  const [pageKey, setPageKey] = useState(0) // For simulating page refresh

  const login = (e) => {
    e.preventDefault()
    if (!username || !password) return

    const sessionId = generateSessionId()
    const maxAge = rememberMe ? 2592000 : null // 30 days or session

    const newSessionCookie = {
      name: '__Host-sessionId',
      value: sessionId,
      attributes: {
        Path: '/',
        Secure: true,
        HttpOnly: true,
        SameSite: 'Strict',
        ...(maxAge && { 'Max-Age': maxAge }),
      },
    }

    setSessionCookie(newSessionCookie)
    setCookies((prev) => [...prev.filter((c) => c.name !== '__Host-sessionId'), newSessionCookie])
    setCurrentUser(username)
    setIsLoggedIn(true)

    // Log HTTP response
    setHttpLog((prev) => [
      ...prev,
      {
        type: 'response',
        status: '200 OK',
        headers: [
          `Set-Cookie: ${newSessionCookie.name}=${newSessionCookie.value}; ${Object.entries(newSessionCookie.attributes)
            .map(([k, v]) => (v === true ? k : `${k}=${v}`))
            .join('; ')}`,
        ],
      },
    ])
  }

  const logout = () => {
    setHttpLog((prev) => [
      ...prev,
      {
        type: 'response',
        status: '200 OK',
        headers: [
          'Set-Cookie: __Host-sessionId=; Path=/; Secure; HttpOnly; Max-Age=0',
        ],
      },
    ])

    setIsLoggedIn(false)
    setCurrentUser(null)
    setSessionCookie(null)
    setUsername('')
    setPassword('')
    setRememberMe(false)
    setCookies((prev) => prev.filter((c) => c.name !== '__Host-sessionId'))
  }

  const addToCart = (product) => {
    const newCart = [...cart, product]
    setCart(newCart)

    const cartValue = newCart.map((p) => p.id).join(',')
    const newCartCookie = {
      name: 'cart',
      value: cartValue,
      attributes: {
        Path: '/',
        'Max-Age': 604800, // 7 days
        SameSite: 'Lax',
      },
    }

    setCartCookie(newCartCookie)
    setCookies((prev) => [...prev.filter((c) => c.name !== 'cart'), newCartCookie])
  }

  const simulateRefresh = () => {
    // In a real scenario, cookies would persist
    // We simulate this by just re-rendering with the same cookie data
    setPageKey((prev) => prev + 1)
    setHttpLog((prev) => [
      ...prev,
      {
        type: 'request',
        method: 'GET',
        url: '/products',
        headers: cookies.map((c) => `Cookie: ${c.name}=${c.value}`),
      },
    ])
  }

  const clearCart = () => {
    setCart([])
    setCartCookie(null)
    setCookies((prev) => prev.filter((c) => c.name !== 'cart'))
  }

  const formatCookieString = (cookie) => {
    if (!cookie) return ''
    const attrs = Object.entries(cookie.attributes)
      .map(([k, v]) => (v === true ? k : `${k}=${v}`))
      .join('; ')
    return `${cookie.name}=${cookie.value}; ${attrs}`
  }

  return (
    <div className="session-demo" key={pageKey}>
      <h2>Session Management</h2>
      <p className="intro">
        See how cookies power authentication and shopping carts. Watch the Set-Cookie headers in real-time.
      </p>

      <div className="session-grid">
        <div className="demo-area">
          <div className="auth-section">
            <h3>🔐 Authentication Demo</h3>

            {!isLoggedIn ? (
              <form className="login-form" onSubmit={login}>
                <div className="form-group">
                  <label htmlFor="username">Username</label>
                  <input
                    id="username"
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Enter any username"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="password">Password</label>
                  <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter any password"
                  />
                </div>
                <label className="remember-me">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                  />
                  Remember me
                </label>
                <button type="submit" className="login-button">
                  Login
                </button>
              </form>
            ) : (
              <div className="user-dashboard">
                <div className="welcome-message">
                  <span className="avatar">👤</span>
                  <span>Welcome, {currentUser}!</span>
                </div>
                <div className="session-info">
                  <p>
                    <strong>Session Type:</strong>{' '}
                    {sessionCookie?.attributes['Max-Age'] ? 'Persistent' : 'Session'}
                  </p>
                  <p className="session-note">
                    {sessionCookie?.attributes['Max-Age']
                      ? '✓ Your session will persist even after closing the browser'
                      : '⚠️ Your session will be lost when you close the browser'}
                  </p>
                </div>
                <button onClick={logout} className="logout-button">
                  Logout
                </button>
              </div>
            )}

            {sessionCookie && (
              <div className="session-cookie-details" data-testid="session-cookie-details">
                <h4>Session Cookie Details</h4>
                <pre><code>{formatCookieString(sessionCookie)}</code></pre>
                <div className="attribute-breakdown">
                  <span className="attr secure">Secure</span>
                  <span className="attr httponly">HttpOnly</span>
                  <span className="attr samesite">SameSite=Strict</span>
                  {sessionCookie.attributes['Max-Age'] && (
                    <span className="attr persistent">Persistent</span>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="shopping-cart-demo" data-testid="shopping-cart-demo">
            <h3>🛒 Shopping Cart Demo</h3>
            <p className="cart-intro">
              Add items to your cart - they're stored in a cookie and persist across "page refreshes"
            </p>

            <div className="products-grid">
              {products.map((product) => (
                <div key={product.id} className="product-item">
                  <span className="product-emoji">{product.emoji}</span>
                  <span className="product-name">{product.name}</span>
                  <span className="product-price">${product.price}</span>
                  <button
                    data-testid="add-item-button"
                    onClick={() => addToCart(product)}
                    className="add-button"
                  >
                    Add
                  </button>
                </div>
              ))}
            </div>

            <div className="cart-section" data-testid="cart-items">
              <h4>Your Cart ({cart.length} items)</h4>
              {cart.length === 0 ? (
                <p className="empty-cart">Your cart is empty</p>
              ) : (
                <>
                  {cart.map((item, idx) => (
                    <div key={idx} className="cart-item" data-testid="cart-item">
                      {item.emoji} {item.name} - ${item.price}
                    </div>
                  ))}
                  <div className="cart-total">
                    Total: ${cart.reduce((sum, item) => sum + item.price, 0)}
                  </div>
                  <button onClick={clearCart} className="clear-cart-button">
                    Clear Cart
                  </button>
                </>
              )}
            </div>

            {cartCookie && (
              <div className="cart-cookie" data-testid="cart-cookie">
                <h4>Cart Cookie</h4>
                <pre><code>{formatCookieString(cartCookie)}</code></pre>
              </div>
            )}

            <button onClick={simulateRefresh} className="refresh-button">
              🔄 Simulate Page Refresh/Navigate
            </button>
          </div>
        </div>

        <div className="cookie-view">
          <div className="cookie-jar" data-testid="cookie-jar">
            <h3>🍪 Cookie Jar</h3>
            {cookies.length === 0 ? (
              <p className="empty-jar">No cookies stored yet</p>
            ) : (
              cookies.map((cookie, idx) => (
                <div key={idx} className="cookie-item" data-testid="cookie-item">
                  <div className="cookie-name">{cookie.name}</div>
                  <div className="cookie-value">{cookie.value}</div>
                  <div className="cookie-attrs">
                    {Object.entries(cookie.attributes).map(([k, v]) => (
                      <span key={k} className="cookie-attr">
                        {v === true ? k : `${k}=${v}`}
                      </span>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="http-viewer" data-testid="http-viewer">
            <h3>📡 HTTP Log</h3>
            {httpLog.length === 0 ? (
              <p className="empty-log">Perform actions to see HTTP traffic</p>
            ) : (
              httpLog.slice(-5).map((entry, idx) => (
                <div key={idx} className={`http-entry ${entry.type}`}>
                  {entry.type === 'response' ? (
                    <>
                      <div className="http-status">{entry.status}</div>
                      {entry.headers.map((h, i) => (
                        <div key={i} className="http-header">{h}</div>
                      ))}
                    </>
                  ) : (
                    <>
                      <div className="http-method">{entry.method} {entry.url}</div>
                      {entry.headers.map((h, i) => (
                        <div key={i} className="http-header">{h}</div>
                      ))}
                    </>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="educational-content">
        <h3>Session vs Persistent Cookies</h3>
        <div className="comparison-table">
          <div className="comparison-row header">
            <div>Feature</div>
            <div>Session Cookie</div>
            <div>Persistent Cookie</div>
          </div>
          <div className="comparison-row">
            <div>Lifetime</div>
            <div>Until browser closes</div>
            <div>Until Max-Age/Expires</div>
          </div>
          <div className="comparison-row">
            <div>Set-Cookie Header</div>
            <div><code>Set-Cookie: id=abc</code></div>
            <div><code>Set-Cookie: id=abc; Max-Age=86400</code></div>
          </div>
          <div className="comparison-row">
            <div>Use Case</div>
            <div>Banking sessions, sensitive actions</div>
            <div>"Remember me", preferences</div>
          </div>
          <div className="comparison-row">
            <div>Security</div>
            <div>More secure (expires quickly)</div>
            <div>Less secure (long-lived)</div>
          </div>
        </div>

        <div className="best-practices">
          <h4>Best Practices for Session Cookies</h4>
          <ul>
            <li><strong>Use __Host- prefix</strong> - Prevents domain override attacks</li>
            <li><strong>Set HttpOnly</strong> - Prevents JavaScript access (XSS protection)</li>
            <li><strong>Set Secure</strong> - Only sent over HTTPS</li>
            <li><strong>Set SameSite=Strict</strong> - Prevents CSRF attacks</li>
            <li><strong>Short expiration</strong> - Minimize exposure window</li>
            <li><strong>Regenerate on login</strong> - Prevents session fixation</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
