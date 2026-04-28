// Configuration
// Change this if your backend runs on a different port
const API_BASE_URL = 'http://localhost:8080';

// State
let uploadedDatasetId = null;
let analysisResult = null;

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const csvFileInput = document.getElementById('csvFile');
const protectedAttrsInput = document.getElementById('protectedAttrs');
const targetColInput = document.getElementById('targetCol');
const predictionColInput = document.getElementById('predictionCol');
const modelNameInput = document.getElementById('modelName');
const uploadBtn = document.getElementById('uploadBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const configError = document.getElementById('configError');
const configSuccess = document.getElementById('configSuccess');
const resultsError = document.getElementById('resultsError');
const resultsSuccess = document.getElementById('resultsSuccess');

const fileInfo = document.getElementById('fileInfo');
const noResults = document.getElementById('noResults');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const metricsDisplay = document.getElementById('metricsDisplay');
const explanationDisplay = document.getElementById('explanationDisplay');

// Event Listeners
uploadBtn.addEventListener('click', handleUpload);
analyzeBtn.addEventListener('click', handleAnalyze);

csvFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileInfo.textContent = `📄 ${file.name} (${sizeMB} MB)`;
        fileInfo.classList.remove('hidden');
    }
});

// Handle CSV Upload
async function handleUpload() {
    clearMessages();

    // Validation
    if (!csvFileInput.files.length) {
        showError(configError, 'Please select a CSV file');
        return;
    }

    if (!targetColInput.value.trim()) {
        showError(configError, 'Please enter a target column name');
        return;
    }

    if (!protectedAttrsInput.value.trim()) {
        showError(configError, 'Please enter at least one protected attribute');
        return;
    }

    // Prepare FormData
    const formData = new FormData();
    formData.append('file', csvFileInput.files[0]);

    // Upload
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Upload failed (${response.status})`);
        }

        const data = await response.json();
        uploadedDatasetId = data.dataset_id;

        showSuccess(
            configSuccess,
            `✓ Dataset uploaded! ID: ${uploadedDatasetId} (${data.rows} rows)`
        );

        analyzeBtn.disabled = false;
        uploadBtn.textContent = 'Upload CSV';
    } catch (error) {
        showError(configError, `Upload error: ${error.message}`);
        uploadBtn.textContent = 'Upload CSV';
    } finally {
        uploadBtn.disabled = false;
    }
}

// Handle Analysis Request
async function handleAnalyze() {
    clearMessages();

    if (!uploadedDatasetId) {
        showError(resultsError, 'Please upload a dataset first');
        return;
    }

    // Parse protected attributes
    const protectedAttrs = protectedAttrsInput.value
        .split(',')
        .map((attr) => attr.trim())
        .filter((attr) => attr.length > 0);

    if (protectedAttrs.length === 0) {
        showError(resultsError, 'Please enter at least one protected attribute');
        return;
    }

    // Prepare request body
    const requestBody = {
        dataset_id: uploadedDatasetId,
        protected_attributes: protectedAttrs,
        target_column: targetColInput.value.trim(),
        prediction_column: predictionColInput.value.trim() || null,
        model_name: modelNameInput.value.trim() || 'unnamed_model',
        skip_anonymization: false,
    };

    // Show loading
    noResults.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
                errorData.detail || `Analysis failed (${response.status})`
            );
        }

        analysisResult = await response.json();
        displayResults(analysisResult);

        showSuccess(resultsSuccess, '✓ Analysis complete!');
    } catch (error) {
        showError(resultsError, `Analysis error: ${error.message}`);
        noResults.classList.remove('hidden');
    } finally {
        loadingSpinner.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Run Analysis';
    }
}

// Display Results
function displayResults(result) {
    metricsDisplay.innerHTML = '';
    explanationDisplay.innerHTML = '';

    // Display fairness metrics by protected attribute
    if (result.per_attribute) {
        Object.entries(result.per_attribute).forEach(
            ([attrName, attrData]) => {
                const card = createMetricCard(attrName, attrData.metrics, attrData.risk_score);
                metricsDisplay.appendChild(card);
            }
        );
    }

    // Display overall summary
    if (result.overall) {
        const overallDiv = document.createElement('div');
        overallDiv.className = 'metric-card';
        const fairnessStatus = result.overall.is_fair ? '✓ Fair' : '⚠️ Unfair';
        overallDiv.innerHTML = `
            <h4>Overall Assessment</h4>
            <p><strong>Status:</strong> ${fairnessStatus}</p>
            <p><strong>Summary:</strong> ${result.overall.summary || 'Analysis complete'}</p>
        `;
        metricsDisplay.appendChild(overallDiv);
    }

    // Display explanation
    if (result.explanation) {
        const explanationDiv = createExplanation(result.explanation);
        explanationDisplay.appendChild(explanationDiv);
    }

    resultsContainer.classList.remove('hidden');
    noResults.classList.add('hidden');
}

// Create Metric Card
function createMetricCard(attrName, metrics, riskScore) {
    const card = document.createElement('div');
    card.className = 'metric-card';

    // Risk level badge
    const riskLevel = riskScore?.risk_level || 'UNKNOWN';
    const riskColor = {
        'LOW': '#4CAF50',
        'MEDIUM': '#FF9800',
        'HIGH': '#F44336'
    }[riskLevel] || '#999';

    let metricsHtml = `<h4>${attrName}</h4>`;
    metricsHtml += `<div style="display: inline-block; background: ${riskColor}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 10px;">Risk: ${riskLevel}</div>`;
    metricsHtml += `<div style="margin-top: 10px;">`;

    // Display metrics
    if (metrics) {
        Object.entries(metrics).forEach(([metricName, metricData]) => {
            const value = typeof metricData === 'object' ? metricData.value : metricData;
            const isFair = typeof metricData === 'object' ? metricData.is_fair : true;
            const fairIcon = isFair ? '✓' : '✗';
            metricsHtml += `<div style="margin: 5px 0;"><strong>${metricName}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value} ${fairIcon}</div>`;
        });
    }

    // Display risk score details
    if (riskScore) {
        metricsHtml += `<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee;">`;
        metricsHtml += `<strong>Risk Score:</strong> ${riskScore.score?.toFixed(2) || 'N/A'}<br>`;
        if (riskScore.weights) {
            metricsHtml += `<small>Weights: ${riskScore.weights.join(', ')}</small>`;
        }
        metricsHtml += `</div>`;
    }

    metricsHtml += `</div>`;
    card.innerHTML = metricsHtml;
    return card;
}

// Create Explanation Section
function createExplanation(explanation) {
    const div = document.createElement('div');
    div.className = 'explanation';

    let html = '<h3>AI Analysis & Insights</h3>';

    if (typeof explanation === 'object') {
        if (explanation.summary) {
            html += `<p><strong>Summary:</strong> ${escapeHtml(explanation.summary)}</p>`;
        }
        if (explanation.findings) {
            html += `<p><strong>Key Findings:</strong></p><p>${escapeHtml(explanation.findings)}</p>`;
        }
        if (explanation.recommendations) {
            html += `<p><strong>Recommendations:</strong></p><p>${escapeHtml(explanation.recommendations)}</p>`;
        }
    } else {
        html += `<p>${escapeHtml(String(explanation))}</p>`;
    }

    div.innerHTML = html;
    return div;
}

// Utility Functions
function formatNumber(num) {
    if (num === null || num === undefined) return 'N/A';
    return typeof num === 'number' ? num.toFixed(3) : String(num);
}

function formatPercent(num) {
    if (num === null || num === undefined) return 'N/A';
    return typeof num === 'number' ? (num * 100).toFixed(2) + '%' : String(num);
}

function getRiskLevel(score) {
    if (score <= 0.3) return 'LOW';
    if (score <= 0.7) return 'MEDIUM';
    return 'HIGH';
}

function showError(element, message) {
    element.textContent = message;
    element.classList.remove('hidden');
}

function showSuccess(element, message) {
    element.textContent = message;
    element.classList.remove('hidden');
}

function clearMessages() {
    configError.classList.add('hidden');
    configSuccess.classList.add('hidden');
    resultsError.classList.add('hidden');
    resultsSuccess.classList.add('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
console.log('EquiLens Frontend loaded. API Base URL:', API_BASE_URL);
