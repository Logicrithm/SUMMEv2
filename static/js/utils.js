/**
 * VidSummarize - Utility Functions
 * Reusable helper functions used throughout the app
 */

const Utils = {
  
  // ========== FILE UTILITIES ==========
  
  /**
   * Format file size to human readable format
   * @param {number} bytes - File size in bytes
   * @returns {string} Formatted file size
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  },

  /**
   * Get file extension from filename
   * @param {string} filename - File name
   * @returns {string} File extension
   */
  getFileExtension(filename) {
    return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
  },

  /**
   * Validate file type
   * @param {File} file - File object
   * @returns {boolean}
   */
  validateFileType(file) {
    return CONFIG.isValidFileFormat(file.type);
  },

  /**
   * Validate file size
   * @param {File} file - File object
   * @returns {boolean}
   */
  validateFileSize(file) {
    return CONFIG.isValidFileSize(file.size);
  },

  // ========== URL UTILITIES ==========
  
  /**
   * Validate YouTube URL
   * @param {string} url - YouTube URL
   * @returns {boolean}
   */
  validateYoutubeUrl(url) {
    if (!url || url.trim() === '') return false;
    return CONFIG.isValidYoutubeUrl(url.trim());
  },

  /**
   * Extract YouTube video ID from URL
   * @param {string} url - YouTube URL
   * @returns {string|null} Video ID or null
   */
  extractYoutubeId(url) {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/,
      /youtube\.com\/embed\/([^&\n?#]+)/
    ];
    
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match && match[1]) return match[1];
    }
    
    return null;
  },

  /**
   * Build query string from object
   * @param {Object} params - Query parameters
   * @returns {string} Query string
   */
  buildQueryString(params) {
    return Object.keys(params)
      .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
      .join('&');
  },

  // ========== TIME UTILITIES ==========
  
  /**
   * Format seconds to MM:SS or HH:MM:SS
   * @param {number} seconds - Time in seconds
   * @returns {string} Formatted time
   */
  formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
  },

  /**
   * Format date to readable string
   * @param {Date} date - Date object
   * @returns {string} Formatted date
   */
  formatDate(date) {
    const options = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(date).toLocaleDateString('en-US', options);
  },

  // ========== STRING UTILITIES ==========
  
  /**
   * Truncate string with ellipsis
   * @param {string} str - String to truncate
   * @param {number} maxLength - Maximum length
   * @returns {string} Truncated string
   */
  truncate(str, maxLength = 50) {
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength) + '...';
  },

  /**
   * Sanitize HTML string
   * @param {string} str - String to sanitize
   * @returns {string} Sanitized string
   */
  sanitizeHtml(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
  },

  /**
   * Generate random ID
   * @param {number} length - ID length
   * @returns {string} Random ID
   */
  generateId(length = 16) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  },

  // ========== DOM UTILITIES ==========
  
  /**
   * Show element with animation
   * @param {HTMLElement} element - Element to show
   */
  showElement(element) {
    element.classList.remove('hidden');
    element.classList.add('animate-fadeIn');
  },

  /**
   * Hide element
   * @param {HTMLElement} element - Element to hide
   */
  hideElement(element) {
    element.classList.add('hidden');
  },

  /**
   * Toggle element visibility
   * @param {HTMLElement} element - Element to toggle
   */
  toggleElement(element) {
    element.classList.toggle('hidden');
  },

  /**
   * Enable element
   * @param {HTMLElement} element - Element to enable
   */
  enableElement(element) {
    element.disabled = false;
    element.classList.remove('disabled');
  },

  /**
   * Disable element
   * @param {HTMLElement} element - Element to disable
   */
  disableElement(element) {
    element.disabled = true;
    element.classList.add('disabled');
  },

  // ========== TOAST NOTIFICATIONS ==========
  
  /**
   * Show toast notification
   * @param {string} type - Toast type (success, error, warning, info)
   * @param {string} message - Toast message
   * @param {number} duration - Duration in milliseconds
   */
  showToast(type, message, duration = CONFIG.UI.TOAST_DURATION) {
    const icons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️'
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div style="display: flex; align-items: center; gap: var(--space-md);">
        <span style="font-size: var(--font-size-xl);">${icons[type]}</span>
        <div>
          <strong>${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
          <p style="margin: 0; font-size: var(--font-size-sm);">${message}</p>
        </div>
      </div>
    `;
    
    document.body.appendChild(toast);

    // Auto remove after duration
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(400px)';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },

  /**
   * Show success toast
   * @param {string} message - Success message
   */
  showSuccess(message) {
    this.showToast('success', message);
  },

  /**
   * Show error toast
   * @param {string} message - Error message
   */
  showError(message) {
    this.showToast('error', message);
  },

  /**
   * Show warning toast
   * @param {string} message - Warning message
   */
  showWarning(message) {
    this.showToast('warning', message);
  },

  /**
   * Show info toast
   * @param {string} message - Info message
   */
  showInfo(message) {
    this.showToast('info', message);
  },

  // ========== LOADING UTILITIES ==========
  
  /**
   * Show loading spinner
   * @param {HTMLElement} container - Container element
   * @param {string} message - Loading message
   */
  showLoader(container, message = 'Processing...') {
    const loader = document.createElement('div');
    loader.className = 'loader-container';
    loader.innerHTML = `
      <div class="loader"></div>
      <p class="text-secondary mt-md">${message}</p>
    `;
    container.appendChild(loader);
  },

  /**
   * Hide loading spinner
   * @param {HTMLElement} container - Container element
   */
  hideLoader(container) {
    const loader = container.querySelector('.loader-container');
    if (loader) loader.remove();
  },

  // ========== DEBOUNCE & THROTTLE ==========
  
  /**
   * Debounce function
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  debounce(func, wait = CONFIG.UI.DEBOUNCE_DELAY) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  /**
   * Throttle function
   * @param {Function} func - Function to throttle
   * @param {number} limit - Time limit in milliseconds
   * @returns {Function} Throttled function
   */
  throttle(func, limit = 1000) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  // ========== VALIDATION UTILITIES ==========
  
  /**
   * Validate email address
   * @param {string} email - Email address
   * @returns {boolean}
   */
  validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  },

  /**
   * Check if value is empty
   * @param {any} value - Value to check
   * @returns {boolean}
   */
  isEmpty(value) {
    return value === null || value === undefined || value === '' || 
           (Array.isArray(value) && value.length === 0) ||
           (typeof value === 'object' && Object.keys(value).length === 0);
  },

  // ========== DOWNLOAD UTILITIES ==========
  
  /**
   * Trigger file download
   * @param {string} url - Download URL
   * @param {string} filename - File name
   */
  downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  },

  /**
   * Download text as file
   * @param {string} content - File content
   * @param {string} filename - File name
   * @param {string} mimeType - MIME type
   */
  downloadTextFile(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    this.downloadFile(url, filename);
    URL.revokeObjectURL(url);
  },

  // ========== COPY TO CLIPBOARD ==========
  
  /**
   * Copy text to clipboard
   * @param {string} text - Text to copy
   * @returns {Promise<boolean>}
   */
  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.showSuccess('Copied to clipboard');
      return true;
    } catch (err) {
      this.showError('Failed to copy to clipboard');
      return false;
    }
  }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Utils;
}