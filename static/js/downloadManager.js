/**
 * VidSummarize - Download Manager
 * Handles results display, previews, and downloads
 */

const DownloadManager = {
  
  // Current job ID
  jobId: null,
  
  // Results data
  resultsData: {
    transcript: '',
    summary: '',
    metadata: {}
  },
  
  // DOM elements
  elements: {},
  
  /**
   * Initialize download manager
   */
  init() {
    this.cacheElements();
    this.getJobId();
    this.attachEventListeners();
    this.loadResults();
    
    CONFIG.log('Download manager initialized', { jobId: this.jobId });
  },
  
  /**
   * Cache DOM elements
   */
  cacheElements() {
    this.elements = {
      loadingState: document.getElementById('loadingState'),
      resultsContent: document.getElementById('resultsContent'),
      emptyState: document.getElementById('emptyState'),
      
      // Video info
      videoDuration: document.getElementById('videoDuration'),
      videoLanguage: document.getElementById('videoLanguage'),
      wordCount: document.getElementById('wordCount'),
      processedDate: document.getElementById('processedDate'),
      
      // Download buttons
      downloadTranscriptPDF: document.getElementById('downloadTranscriptPDF'),
      downloadTranscriptTXT: document.getElementById('downloadTranscriptTXT'),
      downloadSummaryPDF: document.getElementById('downloadSummaryPDF'),
      downloadSummaryTXT: document.getElementById('downloadSummaryTXT'),
      downloadAllBtn: document.getElementById('downloadAllBtn'),
      
      // Preview
      previewTabs: document.querySelectorAll('.preview-tab'),
      transcriptText: document.getElementById('transcriptText'),
      summaryText: document.getElementById('summaryText'),
      copyTranscript: document.getElementById('copyTranscript'),
      copySummary: document.getElementById('copySummary'),
      
      // Stats
      transcriptWords: document.getElementById('transcriptWords'),
      summaryWords: document.getElementById('summaryWords'),
      compressionRatio: document.getElementById('compressionRatio'),
      processingTime: document.getElementById('processingTime'),
      
      // Share
      copyLinkBtn: document.getElementById('copyLinkBtn'),
      shareEmailBtn: document.getElementById('shareEmailBtn'),
      shareTwitterBtn: document.getElementById('shareTwitterBtn')
    };
  },
  
  /**
   * Get job ID from URL
   */
  getJobId() {
    const urlParams = new URLSearchParams(window.location.search);
    this.jobId = urlParams.get('job_id');
    
    if (!this.jobId) {
      const jobIdInput = document.getElementById('jobId');
      if (jobIdInput) {
        this.jobId = jobIdInput.value;
      }
    }
    
    if (!this.jobId) {
      this.showEmptyState();
      return;
    }
  },
  
  /**
   * Attach event listeners
   */
  attachEventListeners() {
    const {
      downloadTranscriptPDF,
      downloadTranscriptTXT,
      downloadSummaryPDF,
      downloadSummaryTXT,
      downloadAllBtn,
      previewTabs,
      copyTranscript,
      copySummary,
      copyLinkBtn,
      shareEmailBtn,
      shareTwitterBtn
    } = this.elements;
    
    // Download buttons
    downloadTranscriptPDF?.addEventListener('click', () => this.downloadFile('transcript', 'pdf'));
    downloadTranscriptTXT?.addEventListener('click', () => this.downloadFile('transcript', 'txt'));
    downloadSummaryPDF?.addEventListener('click', () => this.downloadFile('summary', 'pdf'));
    downloadSummaryTXT?.addEventListener('click', () => this.downloadFile('summary', 'txt'));
    downloadAllBtn?.addEventListener('click', () => this.downloadAll());
    
    // Preview tabs
    previewTabs.forEach(tab => {
      tab.addEventListener('click', () => this.switchTab(tab));
    });
    
    // Copy buttons
    copyTranscript?.addEventListener('click', () => this.copyToClipboard('transcript'));
    copySummary?.addEventListener('click', () => this.copyToClipboard('summary'));
    
    // Share buttons
    copyLinkBtn?.addEventListener('click', () => this.copyLink());
    shareEmailBtn?.addEventListener('click', () => this.shareViaEmail());
    shareTwitterBtn?.addEventListener('click', () => this.shareViaTwitter());
  },
  
  /**
   * Load results from API
   */
  async loadResults() {
    if (!this.jobId) return;
    
    try {
      CONFIG.log('Loading results', { jobId: this.jobId });
      
      // Check job status to get results
      const status = await API.checkStatus(this.jobId);
      
      if (!status.success) {
        throw new Error('Failed to load job status');
      }
      
      if (status.stage !== CONFIG.PROCESSING.STAGES.COMPLETED) {
        throw new Error('Processing not completed or job failed');
      }
      
      // Fetch actual transcript and summary from download endpoints
      await this.fetchTranscriptAndSummary();
      
      // Store metadata
      this.resultsData.metadata = {
        duration: status.duration || 0,
        language: status.language || 'en',
        wordCount: status.word_count || 0,
        processedAt: new Date()
      };
      
      this.displayResults();
      
    } catch (error) {
      CONFIG.error('Failed to load results', error);
      Utils.showError('Failed to load results');
      this.showEmptyState();
    }
  },
  
  /**
   * Fetch actual transcript and summary text
   */
  async fetchTranscriptAndSummary() {
    try {
      // Fetch transcript
      const transcriptResponse = await fetch(`/api/download/transcript/${this.jobId}?format=txt`);
      if (transcriptResponse.ok) {
        const transcriptBlob = await transcriptResponse.blob();
        this.resultsData.transcript = await transcriptBlob.text();
      } else {
        this.resultsData.transcript = 'Transcript not available';
      }
      
      // Fetch summary
      const summaryResponse = await fetch(`/api/download/summary/${this.jobId}?format=txt`);
      if (summaryResponse.ok) {
        const summaryBlob = await summaryResponse.blob();
        this.resultsData.summary = await summaryBlob.text();
      } else {
        this.resultsData.summary = 'Summary not available';
      }
      
      CONFIG.log('Transcript and summary loaded successfully');
      
    } catch (error) {
      CONFIG.error('Failed to fetch transcript/summary', error);
      this.resultsData.transcript = 'Error loading transcript';
      this.resultsData.summary = 'Error loading summary';
    }
  },
  
  /**
   * Display results on page
   */
  displayResults() {
    const { loadingState, resultsContent } = this.elements;
    
    // Hide loading, show content
    Utils.hideElement(loadingState);
    Utils.showElement(resultsContent);
    
    // Populate video info
    this.updateVideoInfo();
    
    // Populate previews
    this.updatePreviews();
    
    // Calculate and display stats
    this.updateStats();
    
    CONFIG.log('Results displayed successfully');
  },
  
  /**
   * Update video information
   */
  updateVideoInfo() {
    const { videoDuration, videoLanguage, wordCount, processedDate } = this.elements;
    const { metadata } = this.resultsData;
    
    if (videoDuration && metadata.duration) {
      videoDuration.textContent = Utils.formatTime(metadata.duration);
    }
    
    if (videoLanguage && metadata.language) {
      videoLanguage.textContent = CONFIG.getLanguageName(metadata.language);
    }
    
    if (wordCount && metadata.wordCount) {
      wordCount.textContent = metadata.wordCount.toLocaleString();
    }
    
    if (processedDate && metadata.processedAt) {
      processedDate.textContent = 'Just now';
    }
  },
  
  /**
   * Update preview sections
   */
  updatePreviews() {
    const { transcriptText, summaryText } = this.elements;
    
    if (transcriptText) {
      transcriptText.textContent = this.resultsData.transcript || 'Loading transcript...';
    }
    
    if (summaryText) {
      summaryText.textContent = this.resultsData.summary || 'Loading summary...';
    }
  },
  
  /**
   * Update statistics
   */
  updateStats() {
    const { transcriptWords, summaryWords, compressionRatio, processingTime } = this.elements;
    
    const transcriptWordCount = this.countWords(this.resultsData.transcript);
    const summaryWordCount = this.countWords(this.resultsData.summary);
    const compression = transcriptWordCount > 0 
      ? Math.round((1 - summaryWordCount / transcriptWordCount) * 100)
      : 0;
    
    if (transcriptWords) {
      transcriptWords.textContent = transcriptWordCount.toLocaleString();
    }
    
    if (summaryWords) {
      summaryWords.textContent = summaryWordCount.toLocaleString();
    }
    
    if (compressionRatio) {
      compressionRatio.textContent = `${compression}%`;
    }
    
    if (processingTime) {
      processingTime.textContent = '~2m';
    }
  },
  
  /**
   * Count words in text
   * @param {string} text - Text to count
   * @returns {number} Word count
   */
  countWords(text) {
    if (!text) return 0;
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  },
  
  /**
   * Switch preview tab
   * @param {HTMLElement} clickedTab - Clicked tab element
   */
  switchTab(clickedTab) {
    const tabName = clickedTab.getAttribute('data-tab');
    
    // Update tab buttons
    this.elements.previewTabs.forEach(tab => {
      tab.classList.remove('active');
    });
    clickedTab.classList.add('active');
    
    // Update content
    document.querySelectorAll('.preview-content').forEach(content => {
      content.classList.remove('active');
    });
    
    const targetContent = document.getElementById(`preview-${tabName}`);
    if (targetContent) {
      targetContent.classList.add('active');
    }
    
    CONFIG.log('Switched to tab', { tab: tabName });
  },
  
  /**
   * Download file
   * @param {string} type - 'transcript' or 'summary'
   * @param {string} format - 'pdf' or 'txt'
   */
  async downloadFile(type, format) {
    if (!this.jobId) return;
    
    CONFIG.log('Downloading file', { type, format, jobId: this.jobId });
    
    try {
      // Show loading state on button
      const button = event.target;
      const originalText = button.innerHTML;
      button.disabled = true;
      button.innerHTML = '<div class="loader loader-sm"></div>';
      
      // Build download URL
      const url = `/api/download/${type}/${this.jobId}?format=${format}`;
      
      // Trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = `${type}_${this.jobId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      Utils.showSuccess(`${type.charAt(0).toUpperCase() + type.slice(1)} downloaded successfully`);
      
      // Restore button
      setTimeout(() => {
        button.disabled = false;
        button.innerHTML = originalText;
      }, 1000);
      
    } catch (error) {
      CONFIG.error('Download failed', error);
      Utils.showError('Download failed. Please try again.');
      
      // Restore button
      if (event.target) {
        event.target.disabled = false;
      }
    }
  },
  
  /**
   * Download all files
   */
  async downloadAll() {
    if (!this.jobId) return;
    
    CONFIG.log('Downloading all files');
    Utils.showInfo('Starting downloads...');
    
    // Download all 4 files with delays
    setTimeout(() => this.downloadFile('transcript', 'pdf'), 100);
    setTimeout(() => this.downloadFile('transcript', 'txt'), 500);
    setTimeout(() => this.downloadFile('summary', 'pdf'), 900);
    setTimeout(() => this.downloadFile('summary', 'txt'), 1300);
    
    setTimeout(() => {
      Utils.showSuccess('All downloads started!');
    }, 1500);
  },
  
  /**
   * Copy text to clipboard
   * @param {string} type - 'transcript' or 'summary'
   */
  async copyToClipboard(type) {
    const text = type === 'transcript' ? this.resultsData.transcript : this.resultsData.summary;
    
    if (!text || text === 'Loading...' || text.includes('not available')) {
      Utils.showError('Nothing to copy');
      return;
    }
    
    try {
      await navigator.clipboard.writeText(text);
      Utils.showSuccess(`${type.charAt(0).toUpperCase() + type.slice(1)} copied to clipboard`);
      CONFIG.log('Copied to clipboard', { type });
    } catch (error) {
      CONFIG.error('Failed to copy', error);
      Utils.showError('Failed to copy to clipboard');
    }
  },
  
  /**
   * Copy results link
   */
  async copyLink() {
    if (!this.jobId) return;
    
    const url = `${window.location.origin}/results?job_id=${this.jobId}`;
    
    try {
      await navigator.clipboard.writeText(url);
      Utils.showSuccess('Link copied to clipboard');
      CONFIG.log('Link copied', { url });
    } catch (error) {
      CONFIG.error('Failed to copy link', error);
      Utils.showError('Failed to copy link');
    }
  },
  
  /**
   * Share via email
   */
  shareViaEmail() {
    if (!this.jobId) return;
    
    const subject = encodeURIComponent('Check out my VidSummarize results');
    const body = encodeURIComponent(
      `I just used VidSummarize to transcribe and summarize a video!\n\n` +
      `View my results: ${window.location.origin}/results?job_id=${this.jobId}`
    );
    
    window.open(`mailto:?subject=${subject}&body=${body}`, '_blank');
    CONFIG.log('Sharing via email');
  },
  
  /**
   * Share via Twitter
   */
  shareViaTwitter() {
    if (!this.jobId) return;
    
    const text = encodeURIComponent(
      'I just used VidSummarize to transcribe and summarize a video! 🎬✨'
    );
    const url = encodeURIComponent(`${window.location.origin}/results?job_id=${this.jobId}`);
    
    window.open(`https://twitter.com/intent/tweet?text=${text}&url=${url}`, '_blank');
    CONFIG.log('Sharing via Twitter');
  },
  
  /**
   * Show empty state
   */
  showEmptyState() {
    const { loadingState, resultsContent, emptyState } = this.elements;
    
    Utils.hideElement(loadingState);
    Utils.hideElement(resultsContent);
    Utils.showElement(emptyState);
    
    CONFIG.log('Showing empty state');
  }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => DownloadManager.init());
} else {
  DownloadManager.init();
}