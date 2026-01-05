import { useState, useEffect } from 'react'
import './TrackingSimulator.css'

const websites = {
  news: {
    name: 'TechNews Daily',
    color: '#3b82f6',
    icon: '📰',
    articles: [
      { id: 'tech1', title: 'AI Revolution in 2026', category: 'Technology' },
      { id: 'pol1', title: 'New Privacy Laws Proposed', category: 'Politics' },
      { id: 'sci1', title: 'Mars Mission Update', category: 'Science' },
    ],
  },
  shopping: {
    name: 'MegaMart Online',
    color: '#22c55e',
    icon: '🛒',
    products: [
      { id: 'laptop', name: 'UltraBook Pro 15"', price: '$1,299', category: 'Electronics' },
      { id: 'headphones', name: 'Noise-Cancel Headphones', price: '$349', category: 'Electronics' },
      { id: 'coffee', name: 'Premium Coffee Maker', price: '$199', category: 'Home' },
    ],
  },
  social: {
    name: 'FriendZone',
    color: '#8b5cf6',
    icon: '👥',
    interests: ['Photography', 'Travel', 'Cooking', 'Gaming'],
  },
}

const adNetworks = ['AdTrack', 'DataHarvest', 'ProfileBuilder']

function generateTrackerId() {
  return 'TRK-' + Math.random().toString(36).substring(2, 10).toUpperCase()
}

export default function TrackingSimulator() {
  const [trackerId] = useState(generateTrackerId)
  const [activeWebsite, setActiveWebsite] = useState(null)
  const [visitHistory, setVisitHistory] = useState([])
  const [interests, setInterests] = useState([])
  const [viewedProducts, setViewedProducts] = useState([])
  const [httpRequests, setHttpRequests] = useState([])
  const [syncAnimating, setSyncAnimating] = useState(false)
  const [retargetedProduct, setRetargetedProduct] = useState(null)

  const visitSite = (siteKey) => {
    setActiveWebsite(siteKey)

    // Add to visit history
    const site = websites[siteKey]
    setVisitHistory((prev) => [...prev, { site: siteKey, time: new Date().toLocaleTimeString() }])

    // Simulate HTTP request with cookie
    const request = {
      id: Date.now(),
      site: site.name,
      type: 'pageview',
      cookie: `_tracker_id=${trackerId}`,
      url: `https://${siteKey}.example.com/`,
    }
    setHttpRequests((prev) => [...prev, request])

    // Trigger cookie sync animation
    setSyncAnimating(true)
    setTimeout(() => setSyncAnimating(false), 2000)
  }

  const interactWithContent = (item, type) => {
    let newInterest = null
    let newProduct = null

    if (type === 'article') {
      newInterest = item.category
    } else if (type === 'product') {
      newInterest = item.category
      newProduct = item
      setViewedProducts((prev) => [...prev, item])
      setRetargetedProduct(item)
    } else if (type === 'interest') {
      newInterest = item
    }

    if (newInterest && !interests.includes(newInterest)) {
      setInterests((prev) => [...prev, newInterest])
    }

    // Log tracking request
    const request = {
      id: Date.now(),
      site: websites[activeWebsite].name,
      type: 'interaction',
      cookie: `_tracker_id=${trackerId}`,
      data: type === 'product' ? `Viewed: ${item.name}` : `Interest: ${newInterest}`,
    }
    setHttpRequests((prev) => [...prev, request])
  }

  const resetTracking = () => {
    setActiveWebsite(null)
    setVisitHistory([])
    setInterests([])
    setViewedProducts([])
    setHttpRequests([])
    setRetargetedProduct(null)
  }

  return (
    <div className="tracking-simulator">
      <h2>Cross-Site Tracking Simulation</h2>
      <p className="intro">
        Experience how third-party cookies track you across websites. Visit different sites and watch your profile being built in real-time.
      </p>

      <p className="explainer">
        <strong>How it works:</strong> A third-party tracking pixel (invisible 1x1 image or JavaScript) is embedded on each website.
        When you visit, the tracker sets a cookie with your unique ID and sends your browsing data back to their servers.
      </p>

      <div className="simulator-grid">
        <div className="websites-panel">
          <h3>Websites</h3>
          <div className="website-tabs">
            {Object.entries(websites).map(([key, site]) => (
              <button
                key={key}
                data-testid={`website-${key}`}
                className={`website-tab ${activeWebsite === key ? 'active' : ''}`}
                style={{ '--site-color': site.color }}
                onClick={() => visitSite(key)}
              >
                <span className="site-icon">{site.icon}</span>
                <span className="site-name">{site.name}</span>
              </button>
            ))}
          </div>

          <div className="website-content">
            {!activeWebsite ? (
              <div className="empty-state">
                <p>Click a website above to start browsing</p>
                <p className="hint">Watch how your activity is tracked across sites</p>
              </div>
            ) : activeWebsite === 'news' ? (
              <div className="news-site">
                <h4>Latest News</h4>
                {websites.news.articles.map((article) => (
                  <div
                    key={article.id}
                    className="article-card"
                    onClick={() => interactWithContent(article, 'article')}
                  >
                    <span className="article-category">{article.category}</span>
                    <h5>{article.title}</h5>
                  </div>
                ))}
                {retargetedProduct && (
                  <div className="retargeted-ad" data-testid="retargeted-ad">
                    <span className="ad-label">Sponsored</span>
                    <p>Still interested in {retargetedProduct.name}?</p>
                    <span className="ad-price">{retargetedProduct.price}</span>
                  </div>
                )}
              </div>
            ) : activeWebsite === 'shopping' ? (
              <div className="shopping-site">
                <h4>Featured Products</h4>
                {websites.shopping.products.map((product) => (
                  <div
                    key={product.id}
                    data-testid={`product-${product.id}`}
                    className="product-card"
                    onClick={() => interactWithContent(product, 'product')}
                  >
                    <div className="product-emoji">
                      {product.id === 'laptop' ? '💻' : product.id === 'headphones' ? '🎧' : '☕'}
                    </div>
                    <h5>{product.name}</h5>
                    <span className="product-price">{product.price}</span>
                  </div>
                ))}
              </div>
            ) : activeWebsite === 'social' ? (
              <div className="social-site">
                <h4>What are you interested in?</h4>
                <div className="interest-chips">
                  {websites.social.interests.map((interest) => (
                    <button
                      key={interest}
                      className="interest-chip"
                      onClick={() => interactWithContent(interest, 'interest')}
                    >
                      {interest}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>

        <div className="tracker-panel" data-testid="tracker-panel">
          <h3>
            <span className="tracker-icon">🕵️</span>
            Ad Tracker's View
          </h3>

          <div className="tracker-id-section">
            <label>Your Tracker ID:</label>
            <code data-testid="tracker-id">{trackerId}</code>
            <p className="id-note">This ID follows you across ALL websites with our tracking pixel</p>
          </div>

          <div className="user-profile">
            <h4>Your Profile (Built from browsing)</h4>

            <div className="profile-section">
              <label>Sites Visited:</label>
              <div className="site-badges">
                {[...new Set(visitHistory.map((v) => v.site))].map((site) => (
                  <span key={site} className="site-badge" style={{ '--site-color': websites[site].color }}>
                    {websites[site].icon} {websites[site].name}
                  </span>
                ))}
              </div>
            </div>

            <div className="profile-section">
              <label>Interests Detected:</label>
              <div className="interests-list" data-testid="interests-list">
                {interests.length === 0 ? (
                  <span className="empty">Browse to reveal interests</span>
                ) : (
                  interests.map((interest) => (
                    <span key={interest} className="interest-tag">
                      {interest}
                    </span>
                  ))
                )}
              </div>
            </div>

            <div className="profile-section">
              <label>Products Viewed:</label>
              <div className="products-viewed">
                {viewedProducts.length === 0 ? (
                  <span className="empty">No products viewed yet</span>
                ) : (
                  viewedProducts.map((product, idx) => (
                    <div key={idx} className="viewed-product">
                      {product.name} - {product.price}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className={`cookie-sync-animation ${syncAnimating ? 'active' : ''}`} data-testid="cookie-sync-animation">
            <h4>Cookie Sync in Progress</h4>
            <div className="sync-networks">
              {adNetworks.map((network, idx) => (
                <div key={network} className="network-node" style={{ '--delay': `${idx * 0.3}s` }}>
                  <span className="network-name">{network}</span>
                  <div className="sync-line" />
                </div>
              ))}
            </div>
            <p className="sync-explanation">
              Ad networks share your ID to build a complete profile across their combined data
            </p>
          </div>

          <button className="reset-button" onClick={resetTracking}>
            🗑️ Clear Browsing Data
          </button>
        </div>
      </div>

      <div className="http-log">
        <h3>HTTP Requests (Tracking Pixels)</h3>
        <div className="requests-list">
          {httpRequests.length === 0 ? (
            <p className="empty">Visit a website to see tracking requests</p>
          ) : (
            httpRequests.slice(-5).map((req) => (
              <div key={req.id} className="http-request" data-testid="http-request">
                <div className="request-header">
                  <span className="method">GET</span>
                  <span className="url">https://tracker.adnetwork.com/pixel.gif</span>
                </div>
                <div className="request-details">
                  <div><strong>Cookie:</strong> {req.cookie}</div>
                  <div><strong>Referer:</strong> {req.url || req.site}</div>
                  {req.data && <div><strong>Data:</strong> {req.data}</div>}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="tracking-explanation">
        <h3>Understanding Third-Party Cookie Tracking</h3>
        <div className="explanation-cards">
          <div className="explanation-card">
            <h4>📍 Tracking Pixel</h4>
            <p>
              An invisible 1x1 image loaded from the tracker's domain. When your browser requests it,
              cookies are automatically sent, allowing the tracker to identify you.
            </p>
          </div>
          <div className="explanation-card">
            <h4>🔗 Cookie Syncing</h4>
            <p>
              Ad networks share user IDs with each other through redirect chains, creating a web
              of interconnected profiles across the entire advertising ecosystem.
            </p>
          </div>
          <div className="explanation-card">
            <h4>🎯 Retargeting</h4>
            <p>
              Products you view on one site follow you across the web as ads. That laptop you looked at?
              It'll appear in ads on news sites, social media, and more.
            </p>
          </div>
          <div className="explanation-card fingerprint">
            <h4>🔍 Fingerprinting</h4>
            <p>
              Even without cookies, trackers can identify you using browser fingerprinting -
              collecting screen size, fonts, GPU info, and more to create a unique identifier.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
