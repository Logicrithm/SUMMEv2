/**
 * VidSummarize - Processing Manager
 * FIXED VERSION - Properly displays word count updates
 */

const ProcessingManager = {
  
  // Current job ID
  jobId: null,
  
  // Polling interval ID
  pollInterval: null,
  
  // Start time for ETA calculation
  startTime: null,
  
  // DOM elements
  elements: {},
  
  /**
   * Initialize processing manager
   */
  init() {
    this.cacheElements();
    this.getJobId();
    this.attachEventListeners();
    this.startProcessing();
    
    CONFIG.log('Processing manager initialized', { jobId: this.jobId });
  },
  
  /**
   * Cache DOM elements
   */
  cacheElements() {
    this.elements = {
      currentStage: document.getElementById('currentStage'),
      statusMessage: document.getElementById('statusMessage'),
      progressBarFill: document.getElementById('progressBarFill'),
      progressPercentage: document.getElementById('progressPercentage'),
      estimatedTime: document.getElementById('estimatedTime'),
      videoInfo: document.getElementById('videoInfo'),
      videoDuration: document.getElementById('videoDuration'),
      videoLanguage: document.getElementById('videoLanguage'),
      videoSize: document.getElementById('videoSize'),
      stagesList: document.getElementById('stagesList'),
      errorContainer: document.getElementById('errorContainer'),
      errorMessage: document.getElementById('errorMessage'),
      successContainer: document.getElementById('successContainer'),
      processingActions: document.getElementById('processingActions'),
      errorActions: document.getElementById('errorActions'),
      successActions: document.getElementById('successActions'),
      cancelBtn: document.getElementById('cancelBtn'),
      retryBtn: document.getElementById('retryBtn'),
      viewResultsBtn: document.getElementById('viewResultsBtn'),
      progressContainer: document.getElementById('progressContainer')
    };
  },
  
  /**
   * Get job ID from URL or hidden input
   */
  getJobId() {
    // Try to get from URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    this.jobId = urlParams.get('job_id');
    
    // If not in URL, try hidden input
    if (!this.jobId) {
      const jobIdInput = document.getElementById('jobId');
      if (jobIdInput) {
        this.jobId = jobIdInput.value;
      }
    }
    
    // If still no job ID, show error
    if (!this.jobId) {
      this.showError('No job ID found. Please start a new upload.');
      return;
    }
  },
  
  /**
   * Attach event listeners
   */
  attachEventListeners() {
    const { cancelBtn, retryBtn, viewResultsBtn } = this.elements;
    
    // Cancel button
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => this.cancelProcessing());
    }
    
    // Retry button
    if (retryBtn) {
      retryBtn.addEventListener('click', () => {
        window.location.href = '/';
      });
    }
    
    // View results button
    if (viewResultsBtn) {
      viewResultsBtn.addEventListener('click', () => {
        window.location.href = `/results?job_id=${this.jobId}`;
      });
    }
  },
  
  /**
   * Start processing and status polling
   */
  async startProcessing() {
    if (!this.jobId) return;
    
    this.startTime = Date.now();
    CONFIG.log('Starting status polling');
    
    try {
      // Start polling status
      await API.pollStatus(this.jobId, (status) => {
        this.updateStatus(status);
      });
      
      // Processing completed
      this.showSuccess();
      
    } catch (error) {
      CONFIG.error('Processing failed', error);
      this.showError(error.message);
    }
  },
  
  /**
   * Update processing status
   * @param {Object} status - Status object from API
   */
  updateStatus(status) {
    CONFIG.log('Status update', status);
    
    const { stage, progress, message, language, duration, word_count, summary_word_count } = status;
    
    // Update current stage text
    if (stage) {
      const stageName = CONFIG.PROCESSING.STAGE_NAMES[stage] || stage;
      this.elements.currentStage.textContent = stageName;
    }
    
    // Update status message with word count if available - FIXED
    if (message) {
      let displayMessage = message;
      
      // Add word count to message if transcription is done
      if (word_count && word_count > 0 && (stage === 'summarizing' || stage === 'generating_output' || stage === 'completed')) {
        displayMessage += ` • ${word_count.toLocaleString()} words transcribed`;
      }
      
      // Add summary word count if summarization is done
      if (summary_word_count && summary_word_count > 0 && (stage === 'generating_output' || stage === 'completed')) {
        displayMessage += ` • ${summary_word_count.toLocaleString()} words in summary`;
      }
      
      this.elements.statusMessage.textContent = displayMessage;
    }
    
    // Update progress bar
    if (progress !== undefined) {
      this.updateProgress(progress);
    }
    
    // Update stage items
    if (stage) {
      this.updateStageItems(stage);
    }
    
    // Update video info - FIXED: Now includes word count
    if (language || duration || word_count) {
      this.updateVideoInfo({ language, duration, word_count, summary_word_count });
    }
    
    // Update estimated time
    this.updateEstimatedTime(progress);
  },
  
  /**
   * Update progress bar
   * @param {number} progress - Progress percentage (0-100)
   */
  updateProgress(progress) {
    const { progressBarFill, progressPercentage } = this.elements;
    
    progressBarFill.style.width = `${progress}%`;
    progressPercentage.textContent = `${Math.round(progress)}%`;
  },
  
  /**
   * Update stage items UI
   * @param {string} currentStage - Current processing stage
   */
  updateStageItems(currentStage) {
    const stages = Object.values(CONFIG.PROCESSING.STAGES);
    const currentIndex = stages.indexOf(currentStage);
    
    stages.forEach((stage, index) => {
      const stageElement = document.getElementById(`stage-${stage}`);
      if (!stageElement) return;
      
      const statusIcon = stageElement.querySelector('.stage-status');
      
      if (index < currentIndex) {
        // Completed stage
        stageElement.classList.remove('active');
        stageElement.classList.add('completed');
        statusIcon.textContent = '✅';
      } else if (index === currentIndex) {
        // Active stage
        stageElement.classList.add('active');
        stageElement.classList.remove('completed');
        statusIcon.textContent = '⏳';
      } else {
        // Pending stage
        stageElement.classList.remove('active', 'completed');
        statusIcon.textContent = '⏳';
      }
    });
  },
  
  /**
   * Update video information - FIXED: Now shows word count
   * @param {Object} info - Video info object
   */
  updateVideoInfo(info) {
    const { videoInfo, videoDuration, videoLanguage, videoSize } = this.elements;
    
    // Show video info section
    Utils.showElement(videoInfo);
    
    // Update duration
    if (info.duration) {
      videoDuration.textContent = Utils.formatTime(info.duration);
    }
    
    // Update language
    if (info.language) {
      videoLanguage.textContent = CONFIG.getLanguageName(info.language);
    }
    
    // FIXED: Update word count display - repurpose "File Size" to show word count
    if (info.word_count && info.word_count > 0) {
      // Change label to "Word Count" if it's still "File Size"
      const sizeLabel = videoInfo.querySelector('.info-item:nth-child(3) .info-label');
      if (sizeLabel && sizeLabel.textContent === 'File Size') {
        sizeLabel.textContent = 'Word Count';
      }
      
      // Display word count
      videoSize.textContent = `${info.word_count.toLocaleString()} words`;
      
      // If summary word count is available, add it
      if (info.summary_word_count && info.summary_word_count > 0) {
        videoSize.textContent = `${info.word_count.toLocaleString()} → ${info.summary_word_count.toLocaleString()} words`;
      }
    }
  },
  
  /**
   * Update estimated time remaining
   * @param {number} progress - Current progress (0-100)
   */
  updateEstimatedTime(progress) {
    if (!this.startTime || progress <= 0) {
      this.elements.estimatedTime.querySelector('span:last-child').textContent = 
        'Estimated time: Calculating...';
      return;
    }
    
    const elapsed = Date.now() - this.startTime;
    const rate = progress / elapsed;
    const remaining = (100 - progress) / rate;
    
    if (remaining > 0 && isFinite(remaining)) {
      const minutes = Math.floor(remaining / 60000);
      const seconds = Math.floor((remaining % 60000) / 1000);
      
      this.elements.estimatedTime.querySelector('span:last-child').textContent = 
        `Estimated time: ${minutes}m ${seconds}s remaining`;
    }
  },
  
  /**
   * Show success state
   */
  showSuccess() {
    CONFIG.log('Processing completed successfully');
    
    const { 
      successContainer, 
      processingActions, 
      successActions,
      progressContainer,
      stagesList 
    } = this.elements;
    
    // Update progress to 100%
    this.updateProgress(100);
    
    // Mark all stages as completed
    this.updateStageItems(CONFIG.PROCESSING.STAGES.COMPLETED);
    
    // Show success message
    Utils.showElement(successContainer);
    Utils.hideElement(processingActions);
    Utils.showElement(successActions);
    
    // Show success toast
    Utils.showSuccess(CONFIG.SUCCESS.PROCESSING_COMPLETE);
    
    // Auto-redirect after 3 seconds
    setTimeout(() => {
      window.location.href = `/results?job_id=${this.jobId}`;
    }, 3000);
  },
  
  /**
   * Show error state
   * @param {string} errorMsg - Error message
   */
  showError(errorMsg) {
    CONFIG.error('Processing error', errorMsg);
    
    const { 
      errorContainer, 
      errorMessage, 
      processingActions, 
      errorActions,
      progressContainer,
      stagesList 
    } = this.elements;
    
    // Update error message
    errorMessage.textContent = errorMsg || CONFIG.ERRORS.PROCESSING_FAILED;
    
    // Show error state
    Utils.showElement(errorContainer);
    Utils.hideElement(processingActions);
    Utils.showElement(errorActions);
    Utils.hideElement(progressContainer);
    
    // Show error toast
    Utils.showError(errorMsg || CONFIG.ERRORS.PROCESSING_FAILED);
  },
  
  /**
   * Cancel processing
   */
  async cancelProcessing() {
    if (!this.jobId) return;
    
    const confirmed = confirm('Are you sure you want to cancel processing?');
    if (!confirmed) return;
    
    CONFIG.log('Cancelling processing', { jobId: this.jobId });
    
    try {
      const result = await API.cancelProcessing(this.jobId);
      
      if (result.success) {
        Utils.showSuccess('Processing cancelled');
        setTimeout(() => {
          window.location.href = '/';
        }, 1500);
      } else {
        throw new Error(result.error);
      }
      
    } catch (error) {
      CONFIG.error('Failed to cancel processing', error);
      Utils.showError('Failed to cancel processing');
    }
  },
  
  /**
   * Stop polling
   */
  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
      CONFIG.log('Stopped status polling');
    }
  }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => ProcessingManager.init());
} else {
  ProcessingManager.init();
}