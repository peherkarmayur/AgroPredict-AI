from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np

# Load the saved ML model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the API
app = FastAPI(title="AgroPredict AI")

# Define the expected JSON input payload
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# The Backend AI Endpoint
@app.post("/predict")
def predict_crop(data: SoilData):
    features = np.array([[
        data.N, data.P, data.K, 
        data.temperature, data.humidity, 
        data.ph, data.rainfall
    ]])
    prediction = model.predict(features)
    return {"status": "success", "recommended_crop": prediction[0]}

# The Frontend Endpoint
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AgroPredict AI | Smart Farming</title>
        <style>
            /* =========================================
               1. CSS VARIABLES & BASE STYLES
               ========================================= */
            :root {
                --dark-green: #145c29;
                --light-green: #27ae60;
                --bg-gradient: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                --text-main: #2c3e50;
                --text-muted: #546e7a;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: system-ui, -apple-system, sans-serif;
            }

            body {
                background: var(--bg-gradient);
                color: var(--text-main);
                line-height: 1.6;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }

            /* =========================================
               2. REUSABLE ANIMATIONS & UTILITIES
               ========================================= */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .fade-in { animation: fadeIn 0.8s ease-out forwards; }
            .hidden { display: none !important; }

            /* Glassmorphism Effect */
            .glass-card {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                padding: 40px;
            }

            /* =========================================
               3. HERO SECTION
               ========================================= */
            header {
                text-align: center;
                padding: 60px 20px;
            }

            h1 {
                font-size: 3rem;
                color: var(--dark-green);
                margin-bottom: 10px;
            }

            .subtitle {
                font-size: 1.2rem;
                color: var(--text-muted);
                margin-bottom: 20px;
            }

            .hero-illustration {
                font-size: 4rem;
                margin-bottom: 20px;
            }

            /* =========================================
               4. STATS & FEATURES (CSS Grid)
               ========================================= */
            .grid-4 {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }

            .card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }

            .card-icon { font-size: 2rem; margin-bottom: 10px; }
            .card h3 { color: var(--dark-green); font-size: 1.1rem; }

            /* =========================================
               5. INPUT FORM (Glassmorphism + Grid)
               ========================================= */
            .form-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }

            /* Desktop 2-column layout */
            @media (min-width: 768px) {
                .form-grid { grid-template-columns: 1fr 1fr; }
                .full-width { grid-column: 1 / -1; }
            }

            .input-group label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: var(--dark-green);
            }

            .input-wrapper {
                display: flex;
                align-items: center;
                background: white;
                border-radius: 10px;
                padding: 10px 15px;
                border: 2px solid transparent;
                transition: border-color 0.3s;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            }

            .input-wrapper:focus-within {
                border-color: var(--light-green);
            }

            .input-wrapper span { margin-right: 10px; font-size: 1.2rem; }
            
            .input-wrapper input {
                border: none;
                outline: none;
                width: 100%;
                font-size: 1rem;
                background: transparent;
            }

            .submit-btn {
                background: var(--light-green);
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 10px;
                cursor: pointer;
                width: 100%;
                transition: background 0.3s;
                box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
            }

            .submit-btn:hover { background: var(--dark-green); }

            /* =========================================
               6. LOADING & RESULT SECTIONS
               ========================================= */
            #loading, #result-box {
                text-align: center;
                margin-top: 30px;
                padding: 30px;
            }

            .spinner {
                border: 4px solid rgba(0,0,0,0.1);
                width: 50px;
                height: 50px;
                border-radius: 50%;
                border-left-color: var(--light-green);
                animation: spin 1s linear infinite;
                margin: 0 auto 15px auto;
            }

            #result-box {
                background: white;
                border: 3px solid var(--light-green);
                border-radius: 15px;
            }

            .result-emoji { font-size: 5rem; margin-bottom: 10px; }
            .result-title { color: var(--dark-green); font-size: 2rem; text-transform: capitalize; }
            
            /* =========================================
               7. FOOTER
               ========================================= */
            footer {
                text-align: center;
                padding: 30px;
                margin-top: auto;
                color: var(--text-muted);
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>

        <header class="fade-in">
            <div class="hero-illustration">🌱🌾🚜</div>
            <h1>AgroPredict AI</h1>
            <p class="subtitle">AI-powered crop recommendation system for smart farming.</p>
        </header>

        <main class="container fade-in">
            
            <!-- Features Section -->
            <section class="grid-4">
                <div class="card"><div class="card-icon">🤖</div><h3>AI Prediction</h3></div>
                <div class="card"><div class="card-icon">🌧</div><h3>Weather Support</h3></div>
                <div class="card"><div class="card-icon">🌱</div><h3>Soil Analysis</h3></div>
                <div class="card"><div class="card-icon">📈</div><h3>Better Yield</h3></div>
            </section>

            <!-- Main Interactive Form Section -->
            <section class="glass-card">
                <form id="prediction-form">
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Nitrogen (N)</label>
                            <div class="input-wrapper"><span>🧪</span><input type="number" id="N" value="90" required></div>
                        </div>
                        <div class="input-group">
                            <label>Phosphorus (P)</label>
                            <div class="input-wrapper"><span>🧫</span><input type="number" id="P" value="42" required></div>
                        </div>
                        <div class="input-group">
                            <label>Potassium (K)</label>
                            <div class="input-wrapper"><span>🔬</span><input type="number" id="K" value="43" required></div>
                        </div>
                        <div class="input-group">
                            <label>Temperature (°C)</label>
                            <div class="input-wrapper"><span>🌡️</span><input type="number" step="0.1" id="temperature" value="20.8" required></div>
                        </div>
                        <div class="input-group">
                            <label>Humidity (%)</label>
                            <div class="input-wrapper"><span>💧</span><input type="number" step="0.1" id="humidity" value="82.0" required></div>
                        </div>
                        <div class="input-group">
                            <label>pH Level</label>
                            <div class="input-wrapper"><span>⚗️</span><input type="number" step="0.1" id="ph" value="6.5" required></div>
                        </div>
                        <div class="input-group full-width">
                            <label>Rainfall (mm)</label>
                            <div class="input-wrapper"><span>🌧️</span><input type="number" step="0.1" id="rainfall" value="202.9" required></div>
                        </div>
                    </div>
                    <button type="submit" class="submit-btn full-width">Analyze Soil Data</button>
                </form>

                <!-- Dynamic Loading State -->
                <div id="loading" class="hidden">
                    <div class="spinner"></div>
                    <h3>Analyzing Soil & Weather Data...</h3>
                </div>

                <!-- Dynamic Result State -->
                <div id="result-box" class="hidden fade-in">
                    <div class="result-emoji" id="crop-emoji">🌾</div>
                    <h2 class="result-title" id="crop-name">Rice</h2>
                    <p>Best crop recommendation based on your localized soil conditions.</p>
                </div>
            </section>

            <!-- Statistics Section -->
            <section class="grid-4" style="margin-top: 40px;">
                <div class="card"><div class="card-icon">🎯</div><h3>95% Accuracy</h3></div>
                <div class="card"><div class="card-icon">🌾</div><h3>22 Crops Supported</h3></div>
                <div class="card"><div class="card-icon">⚙️</div><h3>AI Powered Engine</h3></div>
                <div class="card"><div class="card-icon">⚡</div><h3>Fast Prediction</h3></div>
            </section>

        </main>

        <footer>
            <p>AgroPredict AI © 2026</p>
            <p>Built using Python, FastAPI, HTML, CSS, JavaScript, and Machine Learning.</p>
        </footer>

        <script>
            /* =========================================
               JAVASCRIPT: Logic & API Integration
               ========================================= */
            
            // Emoji dictionary to map text responses to visuals
            const cropEmojis = {
                rice: '🌾', maize: '🌽', chickpea: '🌿', kidneybeans: '🫘',
                pigeonpeas: '🌱', mothbeans: '☘️', mungbean: '🌱', blackgram: '🪴',
                lentil: '🌿', pomegranate: '🍎', banana: '🍌', mango: '🥭',
                grapes: '🍇', watermelon: '🍉', muskmelon: '🍈', apple: '🍏',
                orange: '🍊', papaya: '🥭', coconut: '🥥', cotton: '☁️',
                jute: '🌾', coffee: '☕'
            };

            document.getElementById('prediction-form').addEventListener('submit', async function(event) {
                // Prevent page refresh on submit
                event.preventDefault(); 

                // Manage UI states (Hide result, show loading)
                document.getElementById('result-box').classList.add('hidden');
                document.getElementById('loading').classList.remove('hidden');

                // Read input values
                const requestData = {
                    N: parseFloat(document.getElementById('N').value),
                    P: parseFloat(document.getElementById('P').value),
                    K: parseFloat(document.getElementById('K').value),
                    temperature: parseFloat(document.getElementById('temperature').value),
                    humidity: parseFloat(document.getElementById('humidity').value),
                    ph: parseFloat(document.getElementById('ph').value),
                    rainfall: parseFloat(document.getElementById('rainfall').value)
                };

                try {
                    // Make Async HTTP POST request to Python backend
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestData)
                    });
                    
                    const resultData = await response.json();
                    const crop = resultData.recommended_crop.toLowerCase();
                    
                    // Added a synthetic 800ms delay to let the loading animation play
                    // (This makes the app feel like it's doing "heavy AI processing")
                    setTimeout(() => {
                        document.getElementById('loading').classList.add('hidden');
                        
                        // Update UI with result
                        document.getElementById('crop-name').innerText = crop;
                        document.getElementById('crop-emoji').innerText = cropEmojis[crop] || '🌱';
                        document.getElementById('result-box').classList.remove('hidden');
                    }, 800);

                } catch (error) {
                    alert('Server Error: Ensure the Python FastAPI server is running.');
                    document.getElementById('loading').classList.add('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content