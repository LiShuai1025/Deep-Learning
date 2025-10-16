import DataLoader from './data-loader.js';
import GRUModel from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isDataLoaded = false;
        this.isModelTrained = false;
        
        this.accuracyChart = null;
        this.timelineChart = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload
        document.getElementById('loadData').addEventListener('click', () => this.handleFileUpload());
        
        // Model training
        document.getElementById('trainModel').addEventListener('click', () => this.handleTraining());
        
        // Model evaluation
        document.getElementById('evaluateModel').addEventListener('click', () => this.handleEvaluation());
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a CSV file');
            return;
        }

        try {
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.innerHTML = 'Loading data...';
            
            const result = await this.dataLoader.loadData(file);
            
            fileInfo.innerHTML = `
                Data loaded successfully!<br>
                Stocks: ${result.symbols.join(', ')}<br>
                Total samples: ${result.sampleCount}<br>
                Training samples: ${result.trainSize}<br>
                Test samples: ${result.testSize}
            `;
            
            this.isDataLoaded = true;
            document.getElementById('trainModel').disabled = false;
            
        } catch (error) {
            alert('Error loading file: ' + error.message);
            console.error(error);
        }
    }

    async handleTraining() {
        if (!this.isDataLoaded) {
            alert('Please load data first');
            return;
        }

        try {
            document.getElementById('trainModel').disabled = true;
            const progress = document.getElementById('trainingProgress');
            
            // Build model
            const inputShape = [12, this.dataLoader.symbols.length * 2]; // 12 days, 20 features (10 stocks × 2 features)
            const outputUnits = this.dataLoader.symbols.length * 3; // 30 outputs (10 stocks × 3 days)
            
            this.model.buildModel(inputShape, outputUnits);
            
            // Train model
            progress.innerHTML = 'Starting training...';
            
            await this.model.train(
                this.dataLoader.X_train, 
                this.dataLoader.y_train,
                this.dataLoader.X_test,
                this.dataLoader.y_test,
                50,
                {
                    onEpochEnd: (epoch, logs) => {
                        progress.innerHTML = 
                            `Epoch: ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${logs.acc.toFixed(4)}`;
                    }
                }
            );
            
            progress.innerHTML = 'Training completed!';
            this.isModelTrained = true;
            document.getElementById('evaluateModel').disabled = false;
            
            // Save model
            await this.model.saveModel();
            
        } catch (error) {
            alert('Error during training: ' + error.message);
            console.error(error);
            document.getElementById('trainModel').disabled = false;
        }
    }

    async handleEvaluation() {
        if (!this.isModelTrained) {
            alert('Please train the model first');
            return;
        }

        try {
            const resultsDiv = document.getElementById('accuracyResults');
            resultsDiv.innerHTML = 'Evaluating model...';
            
            // Make predictions
            const predictions = this.model.predict(this.dataLoader.X_test);
            
            // Calculate overall accuracy
            const evaluation = this.model.evaluate(this.dataLoader.X_test, this.dataLoader.y_test);
            
            // Calculate per-stock accuracy
            const stockAccuracies = this.model.calculatePerStockAccuracy(
                this.dataLoader.y_test, 
                predictions, 
                this.dataLoader.symbols
            );
            
            // Sort stocks by accuracy
            const sortedStocks = Object.entries(stockAccuracies)
                .sort(([,a], [,b]) => b - a)
                .reduce((acc, [symbol, accuracy]) => {
                    acc[symbol] = accuracy;
                    return acc;
                }, {});
            
            // Display results
            resultsDiv.innerHTML = `
                Overall Test Accuracy: ${(evaluation.accuracy * 100).toFixed(2)}%<br>
                Test Loss: ${evaluation.loss.toFixed(4)}
            `;
            
            // Create accuracy chart
            this.createAccuracyChart(sortedStocks);
            
            // Create timeline chart for top stock
            this.createTimelineChart(predictions, Object.keys(sortedStocks)[0]);
            
            // Clean up
            predictions.dispose();
            
        } catch (error) {
            alert('Error during evaluation: ' + error.message);
            console.error(error);
        }
    }

    createAccuracyChart(stockAccuracies) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }
        
        const symbols = Object.keys(stockAccuracies);
        const accuracies = Object.values(stockAccuracies).map(acc => acc * 100);
        
        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: symbols,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: accuracies,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Prediction Accuracy (Sorted)'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }

    createTimelineChart(predictions, topStock) {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        
        if (this.timelineChart) {
            this.timelineChart.destroy();
        }
        
        // For simplicity, show predictions for first 20 test samples for the top stock
        const stockIndex = this.dataLoader.symbols.indexOf(topStock);
        const predData = predictions.arraySync();
        const trueData = this.dataLoader.y_test.arraySync();
        
        const sampleCount = Math.min(20, predData.length);
        const labels = Array.from({length: sampleCount}, (_, i) => `Sample ${i + 1}`);
        
        const correctPredictions = [];
        const incorrectPredictions = [];
        
        for (let sample = 0; sample < sampleCount; sample++) {
            let correct = 0;
            let total = 0;
            
            for (let day = 0; day < 3; day++) {
                const outputIndex = day * this.dataLoader.symbols.length + stockIndex;
                const trueVal = trueData[sample][outputIndex];
                const predVal = predData[sample][outputIndex] > 0.5 ? 1 : 0;
                
                if (trueVal === predVal) {
                    correct++;
                }
                total++;
            }
            
            const accuracy = correct / total;
            correctPredictions.push(accuracy * 100);
            incorrectPredictions.push((1 - accuracy) * 100);
        }
        
        this.timelineChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Correct Predictions',
                        data: correctPredictions,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)'
                    },
                    {
                        label: 'Incorrect Predictions',
                        data: incorrectPredictions,
                        backgroundColor: 'rgba(255, 99, 132, 0.6)'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Prediction Results for ${topStock} (First ${sampleCount} Samples)`
                    }
                },
                scales: {
                    x: {
                        stacked: true
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }

    // Clean up when done
    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
