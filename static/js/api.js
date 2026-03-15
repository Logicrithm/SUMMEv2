/**
 * VidSummarize - API Communication Layer
 * All Flask backend API calls centralized here
 */

const API = {
  
  // ========== UPLOAD VIDEO FILE ==========
  
  /**
   * Upload video file to server
   * @param {File} file - Video file object
   * @param {Function} onProgress - Progress callback (percent)
   * @param {string|null} language - Optional language code (e.g., "hi")
   * @returns {Promise<Object>} Response with job_id
   */
  async uploadVideo(file, onProgress = null, language = null) {
    try {
      CONFIG.log('Uploading video file', { name: file.name, size: file.size });
      
      const formData = new FormData();
      formData.append('video', file);
      if (language) {
        formData.append('language', language);
      }
      
      const response = await this._fetchWithProgress(
        CONFIG.getApiUrl('UPLOAD_VIDEO'),
        {
          method: 'POST',
          body: formData
        },
        onProgress
      );
      
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      
      const data = await response.json();
      CONFIG.log('Video uploaded successfully', data);
      
      return {
        success: true,
        jobId: data.job_id,
        message: data.message || 'Upload successful'
      };
      
    } catch (error) {
      CONFIG.error('Video upload failed', error);
      return {
        success: false,
        error: error.message || CONFIG.ERRORS.UPLOAD_FAILED
      };
    }
  },

  // ========== PROCESS YOUTUBE URL ==========
  
  /**
   * Submit YouTube URL for processing
   * @param {string} url - YouTube video URL
   * @param {string|null} language - Optional language code (e.g., "hi")
   * @returns {Promise<Object>} Response with job_id
   */
  async processYoutubeUrl(url, language = null) {
    try {
      CONFIG.log('Processing YouTube URL', { url });
      
      const payload = { url };
      if (language) {
        payload.language = language;
      }

      const response = await fetch(CONFIG.getApiUrl('PROCESS_YOUTUBE'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || response.statusText);
      }
      
      const data = await response.json();
      CONFIG.log('YouTube URL submitted successfully', data);
      
      return {
        success: true,
        jobId: data.job_id,
        message: data.message || 'Processing started'
      };
      
    } catch (error) {
      CONFIG.error('YouTube processing failed', error);
      return {
        success: false,
        error: error.message || CONFIG.ERRORS.PROCESSING_FAILED
      };
    }
  },

  // ========== CHECK PROCESSING STATUS ==========
  
  /**
   * Check processing status for a job
   * @param {string} jobId - Job ID
   * @returns {Promise<Object>} Job status and details
   */
  async checkStatus(jobId) {
    try {
      const response = await fetch(`${CONFIG.getApiUrl('CHECK_STATUS')}/${jobId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      
      const data = await response.json();
      
      return {
        success: true,
        status: data.status,
        stage: data.stage,
        progress: data.progress || 0,
        message: data.message || '',
        language: data.language || null,
        duration: data.duration || null,
        error: data.error || null
      };
      
    } catch (error) {
      CONFIG.error('Status check failed', error);
      return {
        success: false,
        error: error.message || CONFIG.ERRORS.NETWORK_ERROR
      };
    }
  },

  // ========== POLL STATUS UNTIL COMPLETE ==========
  
  /**
   * Poll status until processing is complete or fails
   * @param {string} jobId - Job ID
   * @param {Function} onStatusUpdate - Callback for status updates
   * @returns {Promise<Object>} Final status
   */
  async pollStatus(jobId, onStatusUpdate = null) {
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        // Check for timeout
        if (Date.now() - startTime > CONFIG.PROCESSING.MAX_PROCESSING_TIME) {
          clearInterval(interval);
          reject(new Error(CONFIG.ERRORS.TIMEOUT));
          return;
        }
        
        const status = await this.checkStatus(jobId);
        
        if (!status.success) {
          clearInterval(interval);
          reject(new Error(status.error));
          return;
        }
        
        // Call update callback
        if (onStatusUpdate) {
          onStatusUpdate(status);
        }
        
        // Check if completed or failed
        if (status.stage === CONFIG.PROCESSING.STAGES.COMPLETED) {
          clearInterval(interval);
          resolve(status);
        } else if (status.stage === CONFIG.PROCESSING.STAGES.FAILED) {
          clearInterval(interval);
          reject(new Error(status.error || CONFIG.ERRORS.PROCESSING_FAILED));
        }
        
      }, CONFIG.PROCESSING.STATUS_CHECK_INTERVAL);
    });
  },

  // ========== DOWNLOAD TRANSCRIPT ==========
  
  /**
   * Download transcript file
   * @param {string} jobId - Job ID
   * @param {string} format - File format (txt or pdf)
   * @returns {Promise<Object>} Download result
   */
  async downloadTranscript(jobId, format = 'txt') {
    try {
      CONFIG.log('Downloading transcript', { jobId, format });
      
      const response = await fetch(
        `${CONFIG.getApiUrl('DOWNLOAD_TRANSCRIPT')}/${jobId}?format=${format}`,
        { method: 'GET' }
      );
      
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      
      const blob = await response.blob();
      const filename = `transcript_${jobId}.${format}`;
      
      // Trigger download
      const url = URL.createObjectURL(blob);
      Utils.downloadFile(url, filename);
      URL.revokeObjectURL(url);
      
      CONFIG.log('Transcript downloaded successfully');
      return { success: true };
      
    } catch (error) {
      CONFIG.error('Transcript download failed', error);
      return {
        success: false,
        error: error.message || 'Download failed'
      };
    }
  },

  // ========== DOWNLOAD SUMMARY ==========
  
  /**
   * Download summary file
   * @param {string} jobId - Job ID
   * @param {string} format - File format (txt or pdf)
   * @returns {Promise<Object>} Download result
   */
  async downloadSummary(jobId, format = 'txt') {
    try {
      CONFIG.log('Downloading summary', { jobId, format });
      
      const response = await fetch(
        `${CONFIG.getApiUrl('DOWNLOAD_SUMMARY')}/${jobId}?format=${format}`,
        { method: 'GET' }
      );
      
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      
      const blob = await response.blob();
      const filename = `summary_${jobId}.${format}`;
      
      // Trigger download
      const url = URL.createObjectURL(blob);
      Utils.downloadFile(url, filename);
      URL.revokeObjectURL(url);
      
      CONFIG.log('Summary downloaded successfully');
      return { success: true };
      
    } catch (error) {
      CONFIG.error('Summary download failed', error);
      return {
        success: false,
        error: error.message || 'Download failed'
      };
    }
  },

  // ========== CANCEL PROCESSING ==========
  
  /**
   * Cancel an ongoing processing job
   * @param {string} jobId - Job ID
   * @returns {Promise<Object>} Cancellation result
   */
  async cancelProcessing(jobId) {
    try {
      CONFIG.log('Cancelling job', { jobId });
      
      const response = await fetch(
        `${CONFIG.getApiUrl('CANCEL_PROCESS')}/${jobId}`,
        { method: 'POST' }
      );
      
      if (!response.ok) {
        throw new Error(response.statusText);
      }
      
      const data = await response.json();
      CONFIG.log('Job cancelled successfully', data);
      
      return {
        success: true,
        message: data.message || 'Job cancelled'
      };
      
    } catch (error) {
      CONFIG.error('Job cancellation failed', error);
      return {
        success: false,
        error: error.message || 'Cancellation failed'
      };
    }
  },

  // ========== FETCH WITH PROGRESS ==========
  
  /**
   * Fetch with upload progress tracking
   * @private
   * @param {string} url - Request URL
   * @param {Object} options - Fetch options
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<Response>}
   */
  async _fetchWithProgress(url, options = {}, onProgress = null) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const percentComplete = (e.loaded / e.total) * 100;
          onProgress(Math.round(percentComplete));
        }
      });
      
      xhr.addEventListener('load', () => {
        resolve({
          ok: xhr.status >= 200 && xhr.status < 300,
          status: xhr.status,
          statusText: xhr.statusText,
          json: async () => JSON.parse(xhr.responseText)
        });
      });
      
      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });
      
      xhr.addEventListener('abort', () => {
        reject(new Error('Upload aborted'));
      });
      
      xhr.open(options.method || 'GET', url);
      
      // Set headers if provided
      if (options.headers) {
        Object.keys(options.headers).forEach(key => {
          xhr.setRequestHeader(key, options.headers[key]);
        });
      }
      
      xhr.send(options.body || null);
    });
  },

  // ========== HEALTH CHECK ==========
  
  /**
   * Check if API is available
   * @returns {Promise<boolean>}
   */
  async healthCheck() {
    try {
      const response = await fetch(`${CONFIG.API.BASE_URL}/health`, {
        method: 'GET'
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
