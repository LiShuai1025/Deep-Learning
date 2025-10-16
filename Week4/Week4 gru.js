class GRUModel {
    constructor() {
        this.model = null;
        this.history = null;
    }

    // Build and compile the model
    buildModel(inputShape, outputUnits) {
        this.model = tf.sequential();
        
        // First GRU layer
        this.model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: inputShape
        }));
        
        // Second GRU layer
        this.model.add(tf.layers.gru({
            units: 32,
            returnSequences: false
        }));
        
        // Dropout for regularization
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer - 30 units for 10 stocks × 3 days
        this.model.add(tf.layers.dense({
            units: outputUnits,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });
        
        return this.model;
    }

    // Train the model
    async train(X_train, y_train, X_test, y_test, epochs = 50, callbacks = {}) {
        if (!this.model) {
            throw new Error('Model must be built before training');
        }

        this.history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: 32,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                    // Memory cleanup
                    await tf.nextFrame();
                }
            }
        });

        return this.history;
    }

    // Evaluate the model
    evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model must be built before evaluation');
        }
        
        const results = this.model.evaluate(X_test, y_test);
        const loss = results[0].dataSync()[0];
        const accuracy = results[1].dataSync()[0];
        
        results.forEach(tensor => tensor.dispose());
        
        return { loss, accuracy };
    }

    // Make predictions
    predict(X) {
        if (!this.model) {
            throw new Error('Model must be built before prediction');
        }
        
        return this.model.predict(X);
    }

    // Calculate per-stock accuracy
    calculatePerStockAccuracy(yTrue, yPred, symbols, daysAhead = 3) {
        const trueData = yTrue.arraySync();
        const predData = yPred.arraySync();
        
        const stockAccuracies = {};
        
        symbols.forEach((symbol, symbolIndex) => {
            let correct = 0;
            let total = 0;
            
            for (let sample = 0; sample < trueData.length; sample++) {
                for (let day = 0; day < daysAhead; day++) {
                    const outputIndex = day * symbols.length + symbolIndex;
                    const trueVal = trueData[sample][outputIndex];
                    const predVal = predData[sample][outputIndex] > 0.5 ? 1 : 0;
                    
                    if (trueVal === predVal) {
                        correct++;
                    }
                    total++;
                }
            }
            
            stockAccuracies[symbol] = correct / total;
        });
        
        return stockAccuracies;
    }

    // Save model weights
    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('indexeddb://stock-prediction-model');
        return saveResult;
    }

    // Load model weights
    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('indexeddb://stock-prediction-model');
            return true;
        } catch (error) {
            console.warn('No saved model found:', error);
            return false;
        }
    }

    // Get model summary
    getModelSummary() {
        if (!this.model) {
            return 'No model built';
        }
        
        let summary = '';
        this.model.layers.forEach(layer => {
            summary += `${layer.name} (${layer.getClassName()}) - Output: ${JSON.stringify(layer.outputShape)}\n`;
        });
        return summary;
    }

    // Dispose model
    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}

export default GRUModel;