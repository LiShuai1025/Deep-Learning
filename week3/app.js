// app.js
class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        this.initializeUI();
    }

    /**
     * Initialize UI by binding event listeners to buttons
     */
    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
    }

    /**
     * Handle loading of training and test data from CSV files
     */
    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            this.updateDataStatus(trainData.count, testData.count);
            this.showStatus('Data loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    /**
     * Handle model training with validation split and visualization
     */
    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            this.showStatus('Starting training...');
            
            // Split training data into train/validation sets (90%/10%)
            const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
                this.trainData.xs, this.trainData.ys, 0.1
            );

            // Create model if it doesn't exist
            if (!this.model) {
                this.model = this.createModel();
                this.updateModelInfo();
            }

            // Train model with tfjs-vis callbacks for real-time visualization
            const startTime = Date.now();
            const history = await this.model.fit(trainXs, trainYs, {
                epochs: 5,
                batchSize: 128,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'val_loss', 'acc', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            });

            // Calculate and display training results
            const duration = (Date.now() - startTime) / 1000;
            const bestValAcc = Math.max(...history.history.val_acc);
            
            this.showStatus(`Training completed in ${duration.toFixed(1)}s. Best val_acc: ${bestValAcc.toFixed(4)}`);
            
            // Clean up temporary tensors to avoid memory leaks
            trainXs.dispose();
            trainYs.dispose();
            valXs.dispose();
            valYs.dispose();
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    /**
     * Evaluate model on test data and display comprehensive metrics
     */
    async onEvaluate() {
        if (!this.model) {
            this.showError('No model available. Please train or load a model first.');
            return;
        }

        if (!this.testData) {
            this.showError('No test data available');
            return;
        }

        try {
            this.showStatus('Evaluating model...');
            
            const testXs = this.testData.xs;
            const testYs = this.testData.ys;
            
            // Get model predictions
            const predictions = this.model.predict(testXs);
            const predictedLabels = predictions.argMax(-1); // Convert probabilities to class labels
            const trueLabels = testYs.argMax(-1); // Convert one-hot to class labels
            
            // Calculate overall accuracy
            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            
            // Create confusion matrix for detailed analysis
            const confusionMatrix = await this.createConfusionMatrix(predictedLabels, trueLabels);
            
            // Display metrics in tfjs-vis visor
            const metricsContainer = { name: 'Test Metrics', tab: 'Evaluation' };
            
            // Show model architecture summary
            tfvis.show.modelSummary(metricsContainer, this.model);
            
            // Show per-class accuracy
            tfvis.show.perClassAccuracy(metricsContainer, 
                { values: this.calculatePerClassAccuracy(confusionMatrix) }, 
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            );
            
            // Show confusion matrix heatmap
            tfvis.render.confusionMatrix(metricsContainer, {
                values: confusionMatrix,
                tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            });
            
            this.showStatus(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
            
            // Clean up tensors
            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    /**
     * Test model on 5 random samples and display results visually
     */
    async onTestFive() {
        if (!this.model || !this.testData) {
            this.showError('Please load both model and test data first');
            return;
        }

        try {
            // Get random batch of 5 test samples
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            // Get model predictions
            const predictions = this.model.predict(batchXs);
            const predictedLabels = predictions.argMax(-1);
            const trueLabels = batchYs.argMax(-1);
            
            // Convert tensors to arrays for display
            const predArray = await predictedLabels.array();
            const trueArray = await trueLabels.array();
            
            // Render preview with images and labels
            this.renderPreview(batchXs, predArray, trueArray, indices);
            
            // Clean up tensors
            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
            batchXs.dispose();
            batchYs.dispose();
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    /**
     * Save model to downloadable files (model.json + weights.bin)
     */
    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-cnn');
            this.showStatus('Model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    /**
     * Load model from user-selected files
     */
    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            
            // Dispose old model if exists to free memory
            if (this.model) {
                this.model.dispose();
            }
            
            // Load model from files
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    /**
     * Reset app state - dispose all tensors and clear UI
     */
    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
    }

    /**
     * Toggle tfjs-vis visor visibility
     */
    toggleVisor() {
        tfvis.visor().toggle();
    }

    /**
     * Create CNN model architecture for MNIST classification
     * @returns {tf.Sequential} Compiled model
     */
    createModel() {
        const model = tf.sequential();
        
        // First convolutional layer
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1] // MNIST images are 28x28 pixels with 1 channel (grayscale)
        }));
        
        // Second convolutional layer
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        // Downsample with max pooling
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 })); // Regularization
        
        // Flatten for dense layers
        model.add(tf.layers.flatten());
        
        // Fully connected layer
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 })); // Regularization
        
        // Output layer with softmax for 10 classes (digits 0-9)
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        // Compile model with appropriate loss and metrics
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy', // Suitable for multi-class classification
            metrics: ['accuracy']
        });
        
        return model;
    }

    /**
     * Calculate accuracy between predicted and true labels
     */
    async calculateAccuracy(predicted, trueLabels) {
        const equals = predicted.equal(trueLabels);
        const accuracy = equals.mean();
        const result = await accuracy.data();
        equals.dispose();
        accuracy.dispose();
        return result[0];
    }

    /**
     * Create confusion matrix from predictions and true labels
     */
    async createConfusionMatrix(predicted, trueLabels) {
        const predArray = await predicted.array();
        const trueArray = await trueLabels.array();
        
        // Initialize 10x10 matrix with zeros
        const matrix = Array(10).fill().map(() => Array(10).fill(0));
        
        // Fill confusion matrix
        for (let i = 0; i < predArray.length; i++) {
            const pred = predArray[i];
            const trueVal = trueArray[i];
            matrix[trueVal][pred]++;
        }
        
        return matrix;
    }

    /**
     * Calculate per-class accuracy from confusion matrix
     */
    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i]; // Diagonal elements are correct predictions
            const total = row.reduce((sum, val) => sum + val, 0); // Sum of row is total samples of that class
            return total > 0 ? correct / total : 0;
        });
    }

    /**
     * Render preview of test images with predictions
     */
    renderPreview(images, predicted, trueLabels, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        // Convert tensor to array for processing
        const imageArray = images.arraySync();
        
        // Create preview item for each image
        imageArray.forEach((image, i) => {
            const item = document.createElement('div');
            item.className = 'preview-item';
            
            const canvas = document.createElement('canvas');
            const label = document.createElement('div');
            
            // Color code based on prediction correctness
            const isCorrect = predicted[i] === trueLabels[i];
            label.className = isCorrect ? 'correct' : 'wrong';
            label.textContent = `Pred: ${predicted[i]} | True: ${trueLabels[i]}`;
            
            // Draw image to canvas
            this.dataLoader.draw28x28ToCanvas(tf.tensor(image), canvas, 4);
            
            item.appendChild(canvas);
            item.appendChild(label);
            container.appendChild(item);
        });
    }

    /**
     * Clear preview container
     */
    clearPreview() {
        document.getElementById('previewContainer').innerHTML = '';
    }

    /**
     * Update data status display with sample counts
     */
    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    /**
     * Update model information display
     */
    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
        if (!this.model) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        // Calculate total parameters in model
        let totalParams = 0;
        this.model.layers.forEach(layer => {
            layer.getWeights().forEach(weight => {
                totalParams += weight.size;
            });
        });
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p>Layers: ${this.model.layers.length}</p>
            <p>Total parameters: ${totalParams.toLocaleString()}</p>
        `;
    }

    /**
     * Add status message to training logs
     */
    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    /**
     * Show error message in logs and console
     */
    showError(message) {
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
