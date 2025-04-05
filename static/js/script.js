document.addEventListener('DOMContentLoaded', function() {
    // Audio recording variables
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let recordingTarget = null;
    
    // Recording status elements
    const enrollRecordingStatus = document.getElementById('enrollRecordingStatus');
    const verifyRecordingStatus = document.getElementById('verifyRecordingStatus');
    
    // Record buttons
    const startEnrollRecording = document.getElementById('startEnrollRecording');
    const startVerifyRecording = document.getElementById('startVerifyRecording');
    
    // Set up recording buttons
    if (startEnrollRecording) {
        startEnrollRecording.addEventListener('click', function() {
            recordingTarget = 'enroll';
            toggleRecording(enrollRecordingStatus, 'enrollAudio');
        });
    }
    
    if (startVerifyRecording) {
        startVerifyRecording.addEventListener('click', function() {
            recordingTarget = 'verify';
            toggleRecording(verifyRecordingStatus, 'verifyAudio');
        });
    }
    
    // Function to toggle recording state
    function toggleRecording(statusElement, inputId) {
        if (!isRecording) {
            startRecording(statusElement, inputId);
        } else {
            stopRecording(statusElement, inputId);
        }
    }
    
    // Function to start recording
    function startRecording(statusElement, inputId) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                statusElement.textContent = "Recording... Click again to stop.";
                statusElement.className = "text-danger";
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = e => {
                    audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioFile = new File([audioBlob], "recording.wav", { type: 'audio/wav' });
                    
                    // Create a FileList-like object
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(audioFile);
                    
                    // Set the file input value
                    document.getElementById(inputId).files = dataTransfer.files;
                    
                    // Create audio element for preview
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = document.createElement('audio');
                    audio.src = audioUrl;
                    audio.controls = true;
                    audio.className = "w-100 mt-2";
                    
                    statusElement.textContent = "Recording complete. Preview:";
                    statusElement.className = "text-success";
                    statusElement.appendChild(audio);
                };
                
                mediaRecorder.start();
                isRecording = true;
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                statusElement.textContent = "Error accessing microphone. Please ensure microphone permissions are granted.";
                statusElement.className = "text-danger";
            });
    }
    
    // Function to stop recording
    function stopRecording(statusElement) {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            
            // Stop all audio tracks
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    // Enroll form submission
    const enrollForm = document.getElementById('enrollForm');
    const enrollResult = document.getElementById('enrollResult');
    
    if (enrollForm) {
        enrollForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const userId = document.getElementById('enrollUserId').value;
            const audioFile = document.getElementById('enrollAudio').files[0];
            
            if (!userId || !audioFile) {
                showResult(enrollResult, 'Please provide both User ID and voice sample.', false);
                return;
            }
            
            const formData = new FormData();
            formData.append('user_id', userId);
            formData.append('audio', audioFile);
            
            // Show loading state
            showResult(enrollResult, 'Processing...', null);
            
            fetch('/api/enroll', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showResult(enrollResult, 'Voice profile enrolled successfully', true);
                    
                    // Add audio download button
                    addAudioDownloadButton(enrollResult, userId);
                } else {
                    showResult(enrollResult, `Error: ${data.error}`, false);
                }
            })
            .catch(error => {
                showResult(enrollResult, `Request failed: ${error}`, false);
            });
        });
    }
    
    // Verify form submission
    const verifyForm = document.getElementById('verifyForm');
    const verifyResult = document.getElementById('verifyResult');
    
    if (verifyForm) {
        verifyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const userId = document.getElementById('verifyUserId').value;
            const audioFile = document.getElementById('verifyAudio').files[0];
            
            if (!userId || !audioFile) {
                showResult(verifyResult, 'Please provide both User ID and voice sample.', false);
                return;
            }
            
            const formData = new FormData();
            formData.append('user_id', userId);
            formData.append('audio', audioFile);
            
            // Show loading state
            showResult(verifyResult, 'Processing...', null);
            
            fetch('/api/verify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Use similarity_percentage if available, otherwise calculate it
                    const similarityPercent = data.similarity_percentage !== undefined ? 
                        data.similarity_percentage : 
                        (data.similarity * 100).toFixed(2);
                    
                    // Use threshold_percentage if available, otherwise calculate it
                    const thresholdPercent = data.threshold_percentage !== undefined ? 
                        data.threshold_percentage : 
                        (data.threshold * 100).toFixed(2);
                    
                    const resultMessage = data.match 
                        ? `Voice verified! Similarity: ${similarityPercent}% (Threshold: ${thresholdPercent}%)` 
                        : `Voice did not match. Similarity: ${similarityPercent}% (Threshold: ${thresholdPercent}%)`;
                    
                    showResult(verifyResult, resultMessage, data.match);
                    
                    // Create detailed results table
                    const detailsTable = document.createElement('table');
                    detailsTable.className = 'table table-sm table-bordered mt-3';
                    
                    // Get interpretation text and appropriate styling
                    const interpretation = data.interpretation || '';
                    let interpretationClass = 'text-info';
                    
                    if (interpretation.includes('Definitely different') || 
                        interpretation.includes('Very likely different')) {
                        interpretationClass = 'text-danger';
                    } else if (interpretation.includes('Possibly different')) {
                        interpretationClass = 'text-warning';
                    } else if (interpretation.includes('Likely same') ||
                              interpretation.includes('Very likely same')) {
                        interpretationClass = 'text-success';
                    }
                    
                    // Table content with new interpretation row
                    detailsTable.innerHTML = `
                        <thead class="table-dark">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Similarity Score</td>
                                <td>${similarityPercent}%</td>
                            </tr>
                            <tr>
                                <td>Threshold</td>
                                <td>${thresholdPercent}%</td>
                            </tr>
                            <tr>
                                <td>Result</td>
                                <td>${data.match ? '<span class="text-success">MATCH</span>' : '<span class="text-danger">NO MATCH</span>'}</td>
                            </tr>
                            <tr>
                                <td>Interpretation</td>
                                <td><span class="${interpretationClass}">${interpretation}</span></td>
                            </tr>
                            <tr>
                                <td>User ID</td>
                                <td>${data.user_id}</td>
                            </tr>
                            ${data.details && data.details.enrolled_at ? `
                            <tr>
                                <td>Enrolled At</td>
                                <td>${new Date(data.details.enrolled_at).toLocaleString()}</td>
                            </tr>` : ''}
                        </tbody>
                    `;
                    verifyResult.appendChild(detailsTable);
                    
                    // Add interpretation scale info
                    if (data.details && data.details.threshold_description) {
                        const scaleInfo = document.createElement('div');
                        scaleInfo.className = 'card mt-3';
                        
                        // Create scale content
                        let scaleContent = `
                            <div class="card-header">
                                <h5>Similarity Scale Interpretation</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">`;
                        
                        // Create visual scale with colors
                        scaleContent += `<div class="col-12 mb-3">
                            <div class="d-flex" style="height: 20px">
                                <div class="flex-grow-1 bg-danger" style="width: 20%"></div>
                                <div class="flex-grow-1 bg-warning" style="width: 20%"></div>
                                <div class="flex-grow-1 bg-info" style="width: 20%"></div>
                                <div class="flex-grow-1 bg-success" style="width: 20%"></div>
                                <div class="flex-grow-1 bg-success" style="width: 20%"></div>
                            </div>
                            <div class="d-flex justify-content-between">
                                <span>0.0</span>
                                <span>0.2</span>
                                <span>0.4</span>
                                <span>0.6</span>
                                <span>0.8</span>
                                <span>1.0</span>
                            </div>
                        </div>`;
                        
                        // Add descriptions
                        scaleContent += `<div class="col-12">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Range</th>
                                        <th>Interpretation</th>
                                    </tr>
                                </thead>
                                <tbody>`;
                        
                        // Add rows for each range
                        Object.entries(data.details.threshold_description).forEach(([range, desc]) => {
                            let textClass = 'text-info';
                            if (range.includes('0.0 - 0.2')) textClass = 'text-danger';
                            else if (range.includes('0.2 - 0.4')) textClass = 'text-danger';
                            else if (range.includes('0.4 - 0.6')) textClass = 'text-warning';
                            else if (range.includes('0.6 - 0.8')) textClass = 'text-success';
                            else if (range.includes('0.8 - 1.0')) textClass = 'text-success';
                            
                            scaleContent += `<tr>
                                <td>${range}</td>
                                <td class="${textClass}">${desc}</td>
                            </tr>`;
                        });
                        
                        scaleContent += `</tbody>
                            </table>
                        </div>`;
                        
                        // Close divs
                        scaleContent += `</div>
                            </div>`;
                        
                        scaleInfo.innerHTML = scaleContent;
                        verifyResult.appendChild(scaleInfo);
                    }
                    
                    // Add audio download button if user exists
                    addAudioDownloadButton(verifyResult, userId);
                } else {
                    showResult(verifyResult, `Error: ${data.error}`, false);
                }
            })
            .catch(error => {
                showResult(verifyResult, `Request failed: ${error}`, false);
            });
        });
    }
    
    // Helper function to display results
    function showResult(element, message, isSuccess) {
        element.innerHTML = '';
        
        const resultBox = document.createElement('div');
        resultBox.className = 'alert';
        
        if (isSuccess === true) {
            resultBox.classList.add('alert-success');
        } else if (isSuccess === false) {
            resultBox.classList.add('alert-danger');
        } else {
            resultBox.classList.add('alert-info');
        }
        
        resultBox.textContent = message;
        element.appendChild(resultBox);
    }
    
    // Helper function to add an audio download button
    function addAudioDownloadButton(element, userId) {
        const downloadContainer = document.createElement('div');
        downloadContainer.className = 'mt-3';
        
        const downloadButton = document.createElement('a');
        downloadButton.className = 'btn btn-outline-info';
        downloadButton.href = `/api/audio/${userId}`;
        downloadButton.textContent = 'Download Voice Sample';
        downloadButton.target = '_blank';
        
        downloadContainer.appendChild(downloadButton);
        element.appendChild(downloadContainer);
    }
});
