/**
 * VidSummarize - Main Application
 * Initializes and coordinates all components
 */

const App = {
  
  /**
   * Initialize application
   */
  init() {
    CONFIG.log('VidSummarize application starting...');
    
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.onDOMReady());
    } else {
      this.onDOMReady();
    }
  },
  
  /**
   * Called when DOM is ready
   */
  onDOMReady() {
    CONFIG.log('DOM ready, initializing components...');
    
    // Initialize upload method toggle
    this.initUploadMethodToggle();
    
    // Check API health
    this.checkApiHealth();
    
    // Add page load animation
    this.addPageLoadAnimation();
    
    CONFIG.log('Application initialized successfully');
  },
  
  /**
   * Initialize upload method toggle (Upload vs YouTube)
   */
  initUploadMethodToggle() {
    const uploadMethodBtn = document.getElementById('uploadMethodBtn');
    const youtubeMethodBtn = document.getElementById('youtubeMethodBtn');
    const uploadContent = document.getElementById('uploadContent');
    const youtubeContent = document.getElementById('youtubeContent');
    
    if (!uploadMethodBtn || !youtubeMethodBtn) {
      CONFIG.log('Upload method buttons not found (not on home page)');
      return;
    }
    
    // Upload method click
    uploadMethodBtn.addEventListener('click', () => {
      this.switchUploadMethod('upload');
    });
    
    // YouTube method click
    youtubeMethodBtn.addEventListener('click', () => {
      this.switchUploadMethod('youtube');
    });
    
    CONFIG.log('Upload method toggle initialized');
  },
  
  /**
   * Switch between upload methods
   * @param {string} method - 'upload' or 'youtube'
   */
  switchUploadMethod(method) {
    const uploadMethodBtn = document.getElementById('uploadMethodBtn');
    const youtubeMethodBtn = document.getElementById('youtubeMethodBtn');
    const uploadContent = document.getElementById('uploadContent');
    const youtubeContent = document.getElementById('youtubeContent');
    
    if (method === 'upload') {
      // Activate upload method
      uploadMethodBtn.classList.add('active');
      youtubeMethodBtn.classList.remove('active');
      uploadContent.classList.add('active');
      youtubeContent.classList.remove('active');
      
      CONFIG.log('Switched to upload method');
    } else {
      // Activate YouTube method
      youtubeMethodBtn.classList.add('active');
      uploadMethodBtn.classList.remove('active');
      youtubeContent.classList.add('active');
      uploadContent.classList.remove('active');
      
      CONFIG.log('Switched to YouTube method');
    }
  },
  
  /**
   * Check API health
   */
  async checkApiHealth() {
    try {
      const isHealthy = await API.healthCheck();
      
      if (isHealthy) {
        CONFIG.log('API is healthy');
      } else {
        CONFIG.error('API health check failed');
        
        // Show warning toast if on home page
        if (window.location.pathname === '/' || window.location.pathname === '/home') {
          Utils.showWarning('API server may be offline. Some features may not work.');
        }
      }
    } catch (error) {
      CONFIG.error('API health check error', error);
    }
  },
  
  /**
   * Add page load animation
   */
  addPageLoadAnimation() {
    // Add fade-in animation to main content
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
      mainContent.classList.add('animate-fadeIn');
    }
    
    // Stagger animations for feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
      card.style.animationDelay = `${index * 0.1}s`;
      card.classList.add('animate-fadeIn');
    });
    
    // Stagger animations for step cards
    const stepCards = document.querySelectorAll('.step-card');
    stepCards.forEach((card, index) => {
      card.style.animationDelay = `${index * 0.15}s`;
      card.classList.add('animate-fadeIn');
    });
  },
  
  /**
   * Get URL parameter
   * @param {string} name - Parameter name
   * @returns {string|null}
   */
  getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
  },
  
  /**
   * Navigate to page
   * @param {string} path - Page path
   * @param {Object} params - URL parameters
   */
  navigateTo(path, params = {}) {
    const queryString = Utils.buildQueryString(params);
    const url = queryString ? `${path}?${queryString}` : path;
    window.location.href = url;
  },
  
  /**
   * Show global loading overlay
   */
  showLoading(message = 'Loading...') {
    // Check if overlay already exists
    let overlay = document.getElementById('globalLoadingOverlay');
    
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'globalLoadingOverlay';
      overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 15, 35, 0.95);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        backdrop-filter: blur(10px);
      `;
      
      overlay.innerHTML = `
        <div class="loader loader-lg"></div>
        <p style="color: var(--color-text-secondary); margin-top: var(--space-xl); font-size: var(--font-size-lg);">
          ${message}
        </p>
      `;
      
      document.body.appendChild(overlay);
    }
  },
  
  /**
   * Hide global loading overlay
   */
  hideLoading() {
    const overlay = document.getElementById('globalLoadingOverlay');
    if (overlay) {
      overlay.remove();
    }
  }
};

// Initialize app
App.init();

// Make App globally available
window.VidSummarizeApp = App;