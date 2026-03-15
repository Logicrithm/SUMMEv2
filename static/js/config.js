/**
 * VidSummarize - Configuration File
 * All API endpoints, constants, and app settings
 */

const CONFIG = {
  // ========== API ENDPOINTS ==========
  API: {
    BASE_URL: '/api',  // Change this for production
    ENDPOINTS: {
      UPLOAD_VIDEO: '/upload',
      PROCESS_YOUTUBE: '/youtube',
      CHECK_STATUS: '/status',
      DOWNLOAD_TRANSCRIPT: '/download/transcript',
      DOWNLOAD_SUMMARY: '/download/summary',
      CANCEL_PROCESS: '/cancel'
    }
  },

  // ========== FILE UPLOAD SETTINGS ==========
  UPLOAD: {
    // Accepted video formats
    ACCEPTED_FORMATS: [
      'video/mp4',
      'video/avi',
      'video/mov',
      'video/wmv',
      'video/flv',
      'video/mkv',
      'video/webm'
    ],
    
    // File extensions for display
    ACCEPTED_EXTENSIONS: ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'],
    
    // Maximum file size (500MB in bytes)
    MAX_FILE_SIZE: 500 * 1024 * 1024,
    
    // Chunk size for large file uploads (5MB)
    CHUNK_SIZE: 5 * 1024 * 1024
  },

  // ========== YOUTUBE SETTINGS ==========
  YOUTUBE: {
    // YouTube URL patterns
    URL_PATTERNS: [
      /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/,
      /^(https?:\/\/)?(www\.)?youtube\.com\/watch\?v=[\w-]+/,
      /^(https?:\/\/)?(www\.)?youtu\.be\/[\w-]+/
    ],
    
    // Maximum video duration (in minutes) - optional constraint
    MAX_DURATION: 180,  // 3 hours
  },

  // ========== PROCESSING SETTINGS ==========
  PROCESSING: {
    // Status check interval (milliseconds)
    STATUS_CHECK_INTERVAL: 2000,  // Check every 2 seconds
    
    // Maximum processing time (milliseconds) - 30 minutes
    MAX_PROCESSING_TIME: 30 * 60 * 1000,
    
    // Processing stages
    STAGES: {
      UPLOADING: 'uploading',
      EXTRACTING_AUDIO: 'extracting_audio',
      TRANSCRIBING: 'transcribing',
      SUMMARIZING: 'summarizing',
      GENERATING_OUTPUT: 'generating_output',
      COMPLETED: 'completed',
      FAILED: 'failed'
    },
    
    // Stage display names
    STAGE_NAMES: {
      uploading: 'Uploading Video',
      extracting_audio: 'Extracting Audio',
      transcribing: 'Transcribing Speech',
      summarizing: 'Generating Summary',
      generating_output: 'Preparing Downloads',
      completed: 'Processing Complete',
      failed: 'Processing Failed'
    }
  },

  // ========== UI SETTINGS ==========
  UI: {
    // Toast notification duration (milliseconds)
    TOAST_DURATION: 3000,
    
    // Animation delays
    ANIMATION_DELAY: 100,
    
    // Debounce delay for input validation
    DEBOUNCE_DELAY: 500,
    
    // Page transitions
    PAGE_TRANSITION_DURATION: 300
  },

  // ========== SUPPORTED LANGUAGES ==========
  LANGUAGES: [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ru', name: 'Russian' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ar', name: 'Arabic' },
    { code: 'hi', name: 'Hindi' },
    // Add more languages as needed
  ],

  // ========== ERROR MESSAGES ==========
  ERRORS: {
    FILE_TOO_LARGE: 'File size exceeds 500MB limit',
    INVALID_FORMAT: 'Invalid file format. Please upload a video file',
    INVALID_YOUTUBE_URL: 'Please enter a valid YouTube URL',
    UPLOAD_FAILED: 'Upload failed. Please try again',
    PROCESSING_FAILED: 'Processing failed. Please try again',
    NETWORK_ERROR: 'Network error. Please check your connection',
    TIMEOUT: 'Processing timeout. Please try with a shorter video',
    UNKNOWN_ERROR: 'An unexpected error occurred'
  },

  // ========== SUCCESS MESSAGES ==========
  SUCCESS: {
    UPLOAD_COMPLETE: 'Video uploaded successfully',
    PROCESSING_COMPLETE: 'Processing completed successfully',
    DOWNLOAD_STARTED: 'Download started'
  },

  // ========== VALIDATION ==========
  VALIDATION: {
    // Minimum video duration (seconds)
    MIN_DURATION: 10,
    
    // Maximum video duration (seconds)
    MAX_DURATION: 10800,  // 3 hours
  },

  // ========== DEVELOPMENT SETTINGS ==========
  DEBUG: {
    // Enable console logging
    ENABLE_LOGGING: true,
    
    // Mock API responses for testing
    MOCK_API: false,
    
    // Simulate processing delay (milliseconds)
    MOCK_PROCESSING_DELAY: 5000
  }
};

// ========== HELPER FUNCTIONS ==========

/**
 * Get full API URL
 * @param {string} endpoint - Endpoint key from CONFIG.API.ENDPOINTS
 * @returns {string} Full API URL
 */
CONFIG.getApiUrl = function(endpoint) {
  return this.API.BASE_URL + this.API.ENDPOINTS[endpoint];
};

/**
 * Check if file format is accepted
 * @param {string} mimeType - File MIME type
 * @returns {boolean}
 */
CONFIG.isValidFileFormat = function(mimeType) {
  return this.UPLOAD.ACCEPTED_FORMATS.includes(mimeType);
};

/**
 * Check if file size is within limit
 * @param {number} size - File size in bytes
 * @returns {boolean}
 */
CONFIG.isValidFileSize = function(size) {
  return size <= this.UPLOAD.MAX_FILE_SIZE;
};

/**
 * Check if YouTube URL is valid
 * @param {string} url - YouTube URL
 * @returns {boolean}
 */
CONFIG.isValidYoutubeUrl = function(url) {
  return this.YOUTUBE.URL_PATTERNS.some(pattern => pattern.test(url));
};

/**
 * Get language name by code
 * @param {string} code - Language code
 * @returns {string} Language name
 */
CONFIG.getLanguageName = function(code) {
  const lang = this.LANGUAGES.find(l => l.code === code);
  return lang ? lang.name : 'Unknown';
};

/**
 * Log debug message
 * @param {string} message - Debug message
 * @param {any} data - Additional data
 */
CONFIG.log = function(message, data = null) {
  if (this.DEBUG.ENABLE_LOGGING) {
    console.log(`[VidSummarize] ${message}`, data || '');
  }
};

/**
 * Log error message
 * @param {string} message - Error message
 * @param {any} error - Error object
 */
CONFIG.error = function(message, error = null) {
  console.error(`[VidSummarize Error] ${message}`, error || '');
};

// Freeze config to prevent modifications
Object.freeze(CONFIG);

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CONFIG;
}