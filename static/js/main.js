document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const resultContainer = document.getElementById('resultContainer');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const errorMessage = document.getElementById('errorMessage');
    const medicalText = document.getElementById('medicalText');
    const generateReportBtn = document.getElementById('generateReport');
    const reportContainer = document.getElementById('reportContainer');

    // Handle click to upload
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#2980b9';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    function handleFile(file) {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid image file (JPG, JPEG, or PNG)');
            return;
        }

        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Upload and predict
        uploadAndPredict(file);
    }

    async function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
                return;
            }

            // Show results
            prediction.textContent = data.prediction;
            confidence.textContent = (data.confidence * 100).toFixed(2);
            resultContainer.style.display = 'block';
            errorMessage.style.display = 'none';

        } catch (error) {
            showError('An error occurred during prediction');
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        resultContainer.style.display = 'none';
    }

    // Add report generation functionality
    async function generateReport() {
        const text = medicalText.value.trim();
        
        if (!text) {
            showError('Please enter medical text to generate a report');
            return;
        }

        try {
            // Show loading state
            generateReportBtn.disabled = true;
            generateReportBtn.textContent = 'Generating...';
            
            console.log('Sending request with text:', text);

            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ medical_text: text })
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Received data:', data);

            if (data.error) {
                throw new Error(data.error);
            }

            // Clear any previous reports
            clearReport();
            
            // Display the new report
            displayReport(data);
            
            // Show the report container and hide error message
            document.getElementById('reportContainer').style.display = 'block';
            document.getElementById('errorMessage').style.display = 'none';

        } catch (error) {
            console.error('Error details:', error);
            showError(`Failed to generate report: ${error.message}`);
        } finally {
            // Reset button state
            generateReportBtn.disabled = false;
            generateReportBtn.textContent = 'Generate Report';
        }
    }

    function clearReport() {
        // Clear all report sections
        document.getElementById('reportSummary').textContent = '';
        document.getElementById('medicalEntities').innerHTML = '';
        document.getElementById('findingsCategories').innerHTML = '';
        document.getElementById('recommendations').innerHTML = '';
    }

    function displayReport(report) {
        try {
            console.log('Displaying report:', report);

            // Display summary
            const summaryElement = document.getElementById('reportSummary');
            summaryElement.textContent = report.summary || 'No summary available';

            // Display medical entities
            const entitiesElement = document.getElementById('medicalEntities');
            entitiesElement.innerHTML = formatEntities(report.medical_terms || {});

            // Display categories
            const categoriesElement = document.getElementById('findingsCategories');
            categoriesElement.innerHTML = formatCategories(report.urgency_levels || {});

            // Display recommendations
            const recommendationsElement = document.getElementById('recommendations');
            recommendationsElement.innerHTML = formatRecommendations(report.recommendations || []);

        } catch (error) {
            console.error('Error in displayReport:', error);
            showError('Error displaying report: ' + error.message);
        }
    }

    function formatEntities(entities) {
        try {
            let html = '<div class="entities-grid">';
            
            for (const [category, items] of Object.entries(entities)) {
                if (items && items.length > 0) {
                    html += `
                        <div class="entity-category">
                            <h5>${category.charAt(0).toUpperCase() + category.slice(1)}</h5>
                            <ul>
                                ${items.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
            }
            
            html += '</div>';
            return html;
        } catch (error) {
            console.error('Error in formatEntities:', error);
            return '<p class="error">Error formatting entities</p>';
        }
    }

    function formatCategories(categories) {
        try {
            let html = '<div class="categories-grid">';
            
            for (const [category, items] of Object.entries(categories)) {
                if (items && items.length > 0) {
                    html += `
                        <div class="category ${category}">
                            <h5>${category.charAt(0).toUpperCase() + category.slice(1)}</h5>
                            <ul>
                                ${items.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                }
            }
            
            html += '</div>';
            return html;
        } catch (error) {
            console.error('Error in formatCategories:', error);
            return '<p class="error">Error formatting categories</p>';
        }
    }

    function formatRecommendations(recommendations) {
        try {
            if (!recommendations || recommendations.length === 0) {
                return '<p>No recommendations available</p>';
            }
            
            return recommendations
                .map(rec => `<li>${rec}</li>`)
                .join('');
        } catch (error) {
            console.error('Error in formatRecommendations:', error);
            return '<p class="error">Error formatting recommendations</p>';
        }
    }

    function showError(message) {
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        document.getElementById('reportContainer').style.display = 'none';
    }

    // Add event listener for report generation
    document.getElementById('generateReport').addEventListener('click', generateReport);
}); 