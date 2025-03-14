// Book Translator - Main JavaScript

// Self-executing function to immediately populate language dropdowns 
// This runs as soon as the script loads, before the DOMContentLoaded event
(function() {
    console.log("Immediate language population attempt");
    // Languages to add
    const languages = [
        {"code": "en", "name": "English"},
        {"code": "ta", "name": "Tamil"},
        {"code": "hi", "name": "Hindi"},
        {"code": "mr", "name": "Marathi"}
    ];
    
    // Try to get the dropdown elements
    setTimeout(function() {
        const inputLang = document.getElementById('input-language');
        const outputLang = document.getElementById('output-language');
        
        if (inputLang && outputLang) {
            console.log("Found language dropdowns on immediate execution");
            
            // Clear existing options
            inputLang.innerHTML = '<option value="auto">Auto-detect</option>';
            outputLang.innerHTML = '';
            
            // Add language options
            languages.forEach(lang => {
                // Input dropdown
                const inOption = document.createElement('option');
                inOption.value = lang.code;
                inOption.textContent = lang.name;
                inputLang.appendChild(inOption);
                
                // Output dropdown
                const outOption = document.createElement('option');
                outOption.value = lang.code;
                outOption.textContent = lang.name;
                outputLang.appendChild(outOption);
            });
            
            // Set default output language
            outputLang.value = 'en';
            console.log("Languages populated immediately");
            
            // Try to add translate button event listener immediately as well
            const translateBtn = document.getElementById('translate-btn');
            if (translateBtn) {
                console.log("Found translate button, adding immediate click listener");
                translateBtn.addEventListener('click', function() {
                    console.log("Translate button clicked (immediate binding)");
                    // Redirect to the DOMContentLoaded handler's translateBtn element
                    document.dispatchEvent(new CustomEvent('translateButtonClicked'));
                });
            }
        } else {
            console.warn("Language dropdowns not found on immediate execution");
        }
    }, 100); // Small delay to ensure DOM is accessible
})();

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded, initializing application");
    
    // Initialize elements
    const translateBtn = document.getElementById('translate-btn');
    const prioritySlider = document.getElementById('priority-slider');
    const priorityLabel = document.getElementById('priority-label');
    
    // Set up priority slider event listener
    if (prioritySlider) {
        console.log("Setting up priority slider listener");
        prioritySlider.addEventListener('input', function() {
            const value = this.value;
            updatePriorityLabel(value);
        });
        
        // Initialize priority label
        updatePriorityLabel(prioritySlider.value);
    } else {
        console.error("Priority slider not found");
    }
    
    // Set up translate button event listener
    if (translateBtn) {
        console.log("Setting up translate button listener");
        translateBtn.addEventListener('click', handleTranslation);
    } else {
        console.error("Translate button not found");
    }
    
    /**
     * Update the priority label based on the slider value
     */
    function updatePriorityLabel(value) {
        console.log(`Updating priority label with value: ${value}`);
        if (!priorityLabel) {
            console.error("Priority label element not found");
            return;
        }
        
        let label = getPriorityLabel(value);
        priorityLabel.textContent = `${label} (${value}%)`;
        
        // Update the badge color based on the priority
        priorityLabel.className = 'badge';
        if (value < 20) {
            priorityLabel.classList.add('bg-info');
        } else if (value < 40) {
            priorityLabel.classList.add('bg-primary');
        } else if (value < 60) {
            priorityLabel.classList.add('bg-secondary');
        } else if (value < 80) {
            priorityLabel.classList.add('bg-primary');
        } else {
            priorityLabel.classList.add('bg-success');
        }
    }
    
    // DOM Elements
    const inputText = document.getElementById('input-text');
    const outputText = document.getElementById('output-text');
    const inputLanguage = document.getElementById('input-language');
    const outputLanguage = document.getElementById('output-language');
    const loadingSpinner = document.getElementById('loading-spinner');
    const correlationContainer = document.getElementById('correlation-container');
    const correlationPercentage = document.getElementById('correlation-percentage');
    const correlationBar = document.getElementById('correlation-bar');
    
    // Priority slider elements
    const priorityValue = document.getElementById('priority-value');
    
    // Setup priority slider if it exists
    if (prioritySlider && priorityValue) {
        console.log("Setting up priority slider");
        
        // Initial value display
        updatePriorityLabel(prioritySlider.value);
        
        // Add event listener for slider changes
        prioritySlider.addEventListener('input', function() {
            updatePriorityLabel(this.value);
        });
    } else {
        console.warn("Priority slider elements not found");
    }
    
    // Debug DOM elements
    console.log("DOM Elements after loading:");
    console.log("- Input Text Element:", inputText);
    console.log("- Output Text Element:", outputText);
    console.log("- Input Language Element:", inputLanguage);
    console.log("- Output Language Element:", outputLanguage);
    console.log("- Translate Button Element:", translateBtn);
    console.log("- Priority Slider Element:", prioritySlider);
    
    // Check if translate button exists
    if (!translateBtn) {
        console.error("CRITICAL ERROR: Translate button not found!");
        alert("An error occurred: Translate button not found. Please refresh the page.");
        return;
    }
    
    // New elements for native quality display
    let nativeQualityContainer = document.getElementById('correlation-container')?.cloneNode(true);
    let nativeQualityPercentage = null;
    let nativeQualityBar = null;
    
    // Create native quality elements if they don't exist
    if (nativeQualityContainer && correlationContainer) {
        nativeQualityContainer.id = 'native-quality-container';
        
        // Update content and IDs
        const title = nativeQualityContainer.querySelector('p');
        if (title) title.innerHTML = 'Native-like Quality: <span id="native-quality-percentage">0</span>%';
        
        nativeQualityPercentage = nativeQualityContainer.querySelector('span');
        if (nativeQualityPercentage) nativeQualityPercentage.id = 'native-quality-percentage';
        
        nativeQualityBar = nativeQualityContainer.querySelector('.progress-bar');
        if (nativeQualityBar) nativeQualityBar.id = 'native-quality-bar';
        
        // Add the container after correlation container
        correlationContainer.parentNode.insertBefore(nativeQualityContainer, correlationContainer.nextSibling);
        
        // Hide it initially
        nativeQualityContainer.classList.add('d-none');
        
        console.log("Native quality display elements created:", {
            container: nativeQualityContainer,
            percentage: nativeQualityPercentage,
            bar: nativeQualityBar
        });
    } else {
        console.warn("Could not create native quality display - correlation container not found");
    }

    // Ensure the language select elements exist
    if (!inputLanguage || !outputLanguage) {
        console.error("Language select elements not found in the DOM!");
        // Try to get elements again
        inputLanguage = document.querySelector('#input-language');
        outputLanguage = document.querySelector('#output-language');
        
        if (!inputLanguage || !outputLanguage) {
            console.error("Still couldn't find language select elements. Check HTML IDs.");
            return;
        }
    }

    // Add a custom event listener for the immediate binding redirect
    document.addEventListener('translateButtonClicked', handleTranslation);
    
    console.log("Event listeners initialized");

    /**
     * Handle translation request
     */
    function handleTranslation() {
        console.log("Translation button clicked");
        
        // Get form values
        const inputText = document.getElementById('input-text').value;
        const inputLanguage = document.getElementById('input-language').value;
        const outputLanguage = document.getElementById('output-language').value;
        const priority = document.getElementById('priority-slider').value;
        
        console.log(`Priority value: ${priority}`);
        console.log(`Translation parameters - From: ${inputLanguage}, To: ${outputLanguage}, Priority: ${priority}`);
        
        // Validation
        if (!inputText.trim()) {
            showAlert('Please enter some text to translate.', 'danger');
            return;
        }
        
        // Show loading state
        const translateBtn = document.getElementById('translate-btn');
        const originalBtnText = translateBtn.innerHTML;
        translateBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Translating...';
        translateBtn.disabled = true;
        
        // Add priority indicator to the UI
        updatePriorityIndicator(priority);
        
        // Reset previous results
        document.getElementById('output-text').value = '';
        if (document.getElementById('correlation-container')) {
            document.getElementById('correlation-container').style.display = 'none';
        }
        if (document.getElementById('native-quality-container')) {
            document.getElementById('native-quality-container').style.display = 'none';
        }
        
        console.log("Sending translation request to server");
        console.time('translationRequest');
        
        // Send request to backend
        fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input_text: inputText,
                input_language: inputLanguage,
                output_language: outputLanguage,
                priority: priority
            })
        })
        .then(response => {
            console.timeEnd('translationRequest');
            console.log(`Response status: ${response.status}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Translation response received:", data);
            
            try {
                // Check if data contains a valid translated_text
                if (!data.translated_text) {
                    console.error("Missing translated_text in response:", data);
                    throw new Error("Translation response missing required data");
                }
                
                // Update UI with translation results
                document.getElementById('output-text').value = data.translated_text;
                
                // Update correlation percentage if element exists
                if (document.getElementById('correlation-container')) {
                    updateCorrelation(data.correlation || 0);
                } else {
                    console.warn("Correlation container not found in DOM");
                }
                
                // Debug the native quality data
                console.log("Native quality data in response:", data.native_quality);
                console.log("Native quality container exists:", !!document.getElementById('native-quality-container'));
                
                // Update native quality if element exists
                if (document.hasOwnProperty('native_quality') && data.native_quality.hasOwnProperty('overall_native_quality')) {
                    console.log("Updating native quality to:", data.native_quality.overall_native_quality);
                    updateNativeQuality(data.native_quality.overall_native_quality);
                    
                    // Add detailed native quality information
                    try {
                        const nativeQualityInfo = document.getElementById('native-quality-info');
                        if (nativeQualityInfo) {
                            nativeQualityInfo.innerHTML = `
                                <div class="mt-2 small">
                                    <div><strong>Priority:</strong> ${data.priority} - ${getPriorityLabel(data.priority)}</div>
                                    <div><strong>Idiomaticity:</strong> ${data.native_quality.idiomaticity_score || 0}%</div>
                                    <div><strong>Cultural References:</strong> ${data.native_quality.cultural_reference_count || 0}</div>
                                    <div><strong>Native Fluency:</strong> ${data.native_quality.native_fluency_estimate || 0}%</div>
                                    <div><strong>Common Phrases:</strong> ${data.native_quality.common_phrase_usage || 0}%</div>
                                    <div class="mt-2"><strong>Raw Scores:</strong></div>
                                    <div>Correlation: ${data.raw_scores.correlation.toFixed(1)}%</div>
                                    <div>Native Quality: ${data.raw_scores.native_quality.toFixed(1)}%</div>
                                </div>
                            `;
                        } else {
                            console.warn("Native quality info element not found");
                        }
                    } catch (error) {
                        console.error("Error adding native quality details:", error);
                    }
                } else {
                    console.warn("Native quality data missing or incomplete:", data);
                }
                
                // Show translation complete alert
                showAlert('Translation completed successfully!', 'success');
            } catch (error) {
                console.error("Error processing translation response:", error);
                showAlert(`Error processing translation: ${error.message}`, 'danger');
            }
        })
        .catch(error => {
            console.error("Translation request failed:", error);
            showAlert(`Translation failed: ${error.message}`, 'danger');
        })
        .finally(() => {
            // Reset button state
            translateBtn.innerHTML = originalBtnText;
            translateBtn.disabled = false;
        });
    }

    function updatePriorityIndicator(priority) {
        console.log("Updating priority indicator with value:", priority);
        
        try {
            // Find or create the priority indicator container
            let indicatorContainer = document.getElementById('priority-indicator');
            if (!indicatorContainer) {
                console.log("Creating new priority indicator");
                indicatorContainer = document.createElement('div');
                indicatorContainer.id = 'priority-indicator';
                indicatorContainer.className = 'alert mt-3';
                
                // Find a good place to insert it
                const formContainer = document.querySelector('.form-container');
                if (formContainer) {
                    formContainer.appendChild(indicatorContainer);
                } else {
                    // Fallback to inserting after the priority slider
                    const prioritySlider = document.getElementById('priority-slider');
                    if (prioritySlider && prioritySlider.parentNode) {
                        prioritySlider.parentNode.parentNode.appendChild(indicatorContainer);
                    }
                }
            }
            
            // Determine indicator type based on priority
            let indicatorType = '';
            let message = '';
            
            if (priority < 20) {
                indicatorType = 'alert-info';
                message = '<strong>Strong Essence Preservation:</strong> Your translation will focus on keeping the meaning and intent of the original text, possibly at the expense of natural language in the target language.';
            } else if (priority < 40) {
                indicatorType = 'alert-info';
                message = '<strong>Moderate Essence Preservation:</strong> Your translation will primarily preserve the meaning of the original, with some attention to natural phrasing.';
            } else if (priority < 60) {
                indicatorType = 'alert-secondary';
                message = '<strong>Balanced Translation:</strong> Your translation will balance preserving the original meaning with natural expression in the target language.';
            } else if (priority < 80) {
                indicatorType = 'alert-success';
                message = '<strong>Moderate Native Quality:</strong> Your translation will prioritize natural expression in the target language while still maintaining the core meaning.';
            } else {
                indicatorType = 'alert-success';
                message = '<strong>Strong Native Quality:</strong> Your translation will focus on sounding natural to native speakers, potentially with more idiomatic expressions and cultural references.';
            }
            
            // Update the indicator
            indicatorContainer.className = 'alert mt-3 ' + indicatorType;
            indicatorContainer.innerHTML = message;
            indicatorContainer.style.display = 'block';
            
        } catch (error) {
            console.error("Error updating priority indicator:", error);
        }
    }

    /**
     * Update correlation UI
     */
    function updateCorrelation(percentage) {
        console.log(`Updating correlation bar to ${percentage}%`);
        
        try {
            // Find or get container elements
            const correlationContainer = document.getElementById('correlation-container');
            const correlationValue = document.getElementById('correlation-value');
            const correlationBar = document.getElementById('correlation-bar');
            
            if (!correlationContainer || !correlationValue || !correlationBar) {
                console.error("Correlation elements not found:", {
                    container: correlationContainer,
                    value: correlationValue,
                    bar: correlationBar
                });
                return;
            }
            
            // Format the percentage value
            const formattedPercentage = parseFloat(percentage).toFixed(1);
            
            // Update the text
            correlationValue.textContent = `${formattedPercentage}%`;
            
            // Update the progress bar
            correlationBar.style.width = `${percentage}%`;
            correlationBar.setAttribute('aria-valuenow', percentage);
            correlationBar.setAttribute('data-value', `${formattedPercentage}%`);
            
            // Add the appropriate class based on the percentage
            correlationBar.classList.remove('low', 'medium', 'high');
            if (percentage < 40) {
                correlationBar.classList.add('low');
            } else if (percentage < 70) {
                correlationBar.classList.add('medium');
            } else {
                correlationBar.classList.add('high');
            }
            
            // Add highlight effect
            correlationBar.classList.add('highlight-change');
            setTimeout(() => {
                correlationBar.classList.remove('highlight-change');
            }, 1500);
            
            // Show the container
            correlationContainer.style.display = 'block';
            
            console.log(`Correlation updated successfully to ${formattedPercentage}%`);
        } catch (error) {
            console.error("Error updating correlation:", error);
        }
    }
    
    /**
     * Update native quality UI
     */
    function updateNativeQuality(percentage) {
        console.log(`Updating native quality bar to ${percentage}%`);
        
        try {
            // Find or get container elements
            const nativeQualityContainer = document.getElementById('native-quality-container');
            const nativeQualityValue = document.getElementById('native-quality-value');
            const nativeQualityBar = document.getElementById('native-quality-bar');
            
            if (!nativeQualityContainer || !nativeQualityValue || !nativeQualityBar) {
                console.error("Native quality elements not found:", {
                    container: nativeQualityContainer,
                    value: nativeQualityValue,
                    bar: nativeQualityBar
                });
                return;
            }
            
            // Format the percentage value
            const formattedPercentage = parseFloat(percentage).toFixed(1);
            
            // Update the text
            nativeQualityValue.textContent = `${formattedPercentage}%`;
            
            // Update the progress bar
            nativeQualityBar.style.width = `${percentage}%`;
            nativeQualityBar.setAttribute('aria-valuenow', percentage);
            nativeQualityBar.setAttribute('data-value', `${formattedPercentage}%`);
            
            // Add the appropriate class based on the percentage
            nativeQualityBar.classList.remove('low', 'medium', 'high');
            if (percentage < 40) {
                nativeQualityBar.classList.add('low');
            } else if (percentage < 70) {
                nativeQualityBar.classList.add('medium');
            } else {
                nativeQualityBar.classList.add('high');
            }
            
            // Add highlight effect
            nativeQualityBar.classList.add('highlight-change');
            setTimeout(() => {
                nativeQualityBar.classList.remove('highlight-change');
            }, 1500);
            
            // Show the container
            nativeQualityContainer.style.display = 'block';
            
            console.log(`Native quality updated successfully to ${formattedPercentage}%`);
        } catch (error) {
            console.error("Error updating native quality:", error);
        }
    }

    /**
     * Display an alert message
     */
    function showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) {
            console.error("Alerts container not found");
            return;
        }
        
        const alertId = `alert-${Date.now()}`;
        const alertHtml = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        alertsContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.classList.remove('show');
                setTimeout(() => alertElement.remove(), 150);
            }
        }, 5000);
    }
});

/**
 * Get a descriptive label for the priority value
 */
function getPriorityLabel(value) {
    const numValue = parseInt(value);
    
    if (numValue < 20) {
        return "Strong Essence";
    } else if (numValue < 40) {
        return "Moderate Essence";
    } else if (numValue < 60) {
        return "Balanced";
    } else if (numValue < 80) {
        return "Moderate Native";
    } else {
        return "Strong Native";
    }
} 