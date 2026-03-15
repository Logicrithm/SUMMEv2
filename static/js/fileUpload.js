/**
 * VidSummarize - File Upload Handler
 * Handles drag & drop, file selection, validation, and upload
 */

const FileUploadHandler = {
  
  // Selected file
  selectedFile: null,
  
  // DOM elements
  elements: {},
  
  /**
   * Initialize file upload handler
   */
  init() {
    this.cacheElements();
    this.attachEventListeners();
    CONFIG.log('File upload handler initialized');
  },
  
  /**
   * Cache DOM elements
   */
  cacheElements() {
    this.elements = {
      fileUploadZone: document.getElementById('fileUploadZone'),
      fileInput: document.getElementById('videoFileInput'),
      fileInfo: document.getElementById('fileInfo'),
      fileName: document.getElementById('fileName'),
      fileSize: document.getElementById('fileSize'),
      removeFileBtn: document.getElementById('removeFileBtn'),
      uploadBtn: document.getElementById('uploadBtn')
    };
  },
  
  /**
   * Attach event listeners
   */
  attachEventListeners() {
    const { fileUploadZone, fileInput, removeFileBtn, uploadBtn } = this.elements;
    
    // File input change
    fileInput.addEventListener('change', (e) => {
      this.handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop events
    fileUploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      fileUploadZone.classList.add('drag-over');
    });
    
    fileUploadZone.addEventListener('dragleave', () => {
      fileUploadZone.classList.remove('drag-over');
    });
    
    fileUploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      fileUploadZone.classList.remove('drag-over');
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileSelect(files[0]);
      }
    });
    
    // Remove file button
    removeFileBtn.addEventListener('click', () => {
      this.clearFileSelection();
    });
    
    // Upload button
    uploadBtn.addEventListener('click', () => {
      this.startUpload();
    });
  },
  
  /**
   * Handle file selection
   * @param {File} file - Selected file
   */
  handleFileSelect(file) {
    CONFIG.log('File selected', { name: file.name, size: file.size, type: file.type });
    
    // Validate file type
    if (!Utils.validateFileType(file)) {
      Utils.showError(CONFIG.ERRORS.INVALID_FORMAT);
      return;
    }
    
    // Validate file size
    if (!Utils.validateFileSize(file)) {
      Utils.showError(CONFIG.ERRORS.FILE_TOO_LARGE);
      return;
    }
    
    // Store selected file
    this.selectedFile = file;
    
    // Update UI
    this.updateFileInfo();
    Utils.showSuccess('File selected successfully');
  },
  
  /**
   * Update file info display
   */
  updateFileInfo() {
    const { fileName, fileSize, fileInfo } = this.elements;
    
    if (this.selectedFile) {
      fileName.textContent = this.selectedFile.name;
      fileSize.textContent = Utils.formatFileSize(this.selectedFile.size);
      Utils.showElement(fileInfo);
    }
  },
  
  /**
   * Clear file selection
   */
  clearFileSelection() {
    this.selectedFile = null;
    this.elements.fileInput.value = '';
    Utils.hideElement(this.elements.fileInfo);
    CONFIG.log('File selection cleared');
  },
  
  /**
   * Start file upload
   */
  async startUpload() {
    if (!this.selectedFile) {
      Utils.showError('Please select a file first');
      return;
    }
    
    const { uploadBtn } = this.elements;
    
    // Disable upload button
    Utils.disableElement(uploadBtn);
    uploadBtn.innerHTML = '<div class="loader loader-sm"></div> Uploading...';
    
    try {
      // Upload file with progress tracking
      const result = await API.uploadVideo(this.selectedFile, (progress) => {
        uploadBtn.textContent = `Uploading... ${progress}%`;
        CONFIG.log('Upload progress', { progress });
      });
      
      if (result.success) {
        CONFIG.log('Upload successful', result);
        Utils.showSuccess(CONFIG.SUCCESS.UPLOAD_COMPLETE);
        
        // Redirect to processing page with job ID
        window.location.href = `/processing?job_id=${result.jobId}`;
      } else {
        throw new Error(result.error);
      }
      
    } catch (error) {
      CONFIG.error('Upload failed', error);
      Utils.showError(error.message || CONFIG.ERRORS.UPLOAD_FAILED);
      
      // Re-enable upload button
      Utils.enableElement(uploadBtn);
      uploadBtn.innerHTML = '🚀 Start Processing';
    }
  },
  
  /**
   * Get selected file
   * @returns {File|null}
   */
  getSelectedFile() {
    return this.selectedFile;
  },
  
  /**
   * Check if file is selected
   * @returns {boolean}
   */
  hasFile() {
    return this.selectedFile !== null;
  }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => FileUploadHandler.init());
} else {
  FileUploadHandler.init();
}