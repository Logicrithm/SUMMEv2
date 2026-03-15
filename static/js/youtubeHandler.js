/**
 * VidSummarize - YouTube URL Handler
 * Handles YouTube URL validation and submission
 */

const YouTubeHandler = {
  
  // DOM elements
  elements: {},
  
  /**
   * Initialize YouTube handler
   */
  init() {
    this.cacheElements();
    this.attachEventListeners();
    CONFIG.log('YouTube handler initialized');
  },
  
  /**
   * Cache DOM elements
   */
  cacheElements() {
    this.elements = {
      youtubeUrlInput: document.getElementById('youtubeUrlInput'),
      youtubeSubmitBtn: document.getElementById('youtubeSubmitBtn')
    };
  },
  
  /**
   * Attach event listeners
   */
  attachEventListeners() {
    const { youtubeUrlInput, youtubeSubmitBtn } = this.elements;
    
    // Real-time URL validation with debounce
    youtubeUrlInput.addEventListener('input', Utils.debounce((e) => {
      this.validateUrl(e.target.value);
    }, CONFIG.UI.DEBOUNCE_DELAY));
    
    // Submit button click
    youtubeSubmitBtn.addEventListener('click', () => {
      this.submitUrl();
    });
    
    // Enter key to submit
    youtubeUrlInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.submitUrl();
      }
    });
  },
  
  /**
   * Validate YouTube URL
   * @param {string} url - YouTube URL
   * @returns {boolean}
   */
  validateUrl(url) {
    const { youtubeUrlInput } = this.elements;
    
    if (!url || url.trim() === '') {
      youtubeUrlInput.style.borderColor = '';
      return false;
    }
    
    const isValid = Utils.validateYoutubeUrl(url);
    
    // Update input border color
    if (isValid) {
      youtubeUrlInput.style.borderColor = 'var(--color-accent-green)';
    } else {
      youtubeUrlInput.style.borderColor = 'var(--color-accent-red)';
    }
    
    return isValid;
  },
  
  /**
   * Submit YouTube URL for processing
   */
  async submitUrl() {
    const { youtubeUrlInput, youtubeSubmitBtn } = this.elements;
    const url = youtubeUrlInput.value.trim();
    
    // Validate URL
    if (!url) {
      Utils.showError('Please enter a YouTube URL');
      return;
    }
    
    if (!this.validateUrl(url)) {
      Utils.showError(CONFIG.ERRORS.INVALID_YOUTUBE_URL);
      return;
    }
    
    // Extract video ID for logging
    const videoId = Utils.extractYoutubeId(url);
    CONFIG.log('Submitting YouTube URL', { url, videoId });
    
    // Disable submit button
    Utils.disableElement(youtubeSubmitBtn);
    youtubeSubmitBtn.innerHTML = '<div class="loader loader-sm"></div> Processing...';
    
    try {
      // Submit URL to API
      const result = await API.processYoutubeUrl(url);
      
      if (result.success) {
        CONFIG.log('YouTube URL submitted successfully', result);
        Utils.showSuccess('Processing started');
        
        // Redirect to processing page with job ID
        window.location.href = `/processing?job_id=${result.jobId}`;
      } else {
        throw new Error(result.error);
      }
      
    } catch (error) {
      CONFIG.error('YouTube URL submission failed', error);
      Utils.showError(error.message || CONFIG.ERRORS.PROCESSING_FAILED);
      
      // Re-enable submit button
      Utils.enableElement(youtubeSubmitBtn);
      youtubeSubmitBtn.innerHTML = '🚀 Process';
    }
  },
  
  /**
   * Get YouTube URL from input
   * @returns {string}
   */
  getUrl() {
    return this.elements.youtubeUrlInput.value.trim();
  },
  
  /**
   * Clear YouTube URL input
   */
  clearUrl() {
    this.elements.youtubeUrlInput.value = '';
    this.elements.youtubeUrlInput.style.borderColor = '';
  }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => YouTubeHandler.init());
} else {
  YouTubeHandler.init();
}