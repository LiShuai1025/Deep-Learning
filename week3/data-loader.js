// data-loader.js
class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    /**
     * Parse CSV file and convert to tensors
     * Each row: first value is label (0-9), next 784 values are pixel data (0-255)
     * Normalizes pixels to [0,1] and reshapes to [N, 28, 28, 1]
     * One-hot encodes labels to depth 10
     */
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    
                    const labels = [];
                    const pixels = [];
                    
                    // Parse each line of CSV
                    for (const line of lines) {
                        const values = line.split(',').map(Number);
                        // Skip lines that don't have exactly 785 values (label + 784 pixels)
                        if (values.length !== 785) continue;
                        
                        labels.push(values[0]); // First value is label
                        pixels.push(values.slice(1)); // Remaining 784 values are pixels
                    }
                    
                    if (labels.length === 0) {
                        reject(new Error('No valid data found in file'));
                        return;
                    }
                    
                    // Create tensors in tidy to automatically clean up intermediates
                    const xs = tf.tidy(() => {
                        return tf.tensor2d(pixels)
                            .div(255) // Normalize to [0,1]
                            .reshape([labels.length, 28, 28, 1]); // Reshape for CNN
                    });
                    
                    // One-hot encode labels
                    const ys = tf.tidy(() => {
                        return tf.oneHot(labels, 10);
                    });
                    
                    resolve({ xs, ys, count: labels.length });
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    /**
     * Split training data into training and validation sets
     * @param {tf.Tensor} xs - Input features tensor
     * @param {tf.Tensor} ys - Labels tensor  
     * @param {number} valRatio - Ratio of data to use for validation (default 10%)
     * @returns {Object} Split datasets
     */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numVal = Math.floor(xs.shape[0] * valRatio);
            const numTrain = xs.shape[0] - numVal;
            
            // Split tensors along the first dimension (samples)
            const trainXs = xs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
            const trainYs = ys.slice([0, 0], [numTrain, 10]);
            
            const valXs = xs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);
            const valYs = ys.slice([numTrain, 0], [numVal, 10]);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    /**
     * Get random batch of test samples for preview
     * @param {tf.Tensor} xs - Test features
     * @param {tf.Tensor} ys - Test labels
     * @param {number} k - Number of samples to return
     * @returns {Object} Batch of samples and their indices
     */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            // Create shuffled indices and take first k
            const shuffledIndices = tf.util.createShuffledIndices(xs.shape[0]);
            const selectedIndices = Array.from(shuffledIndices.slice(0, k));
            
            // Gather selected samples
            const batchXs = tf.gather(xs, selectedIndices);
            const batchYs = tf.gather(ys, selectedIndices);
            
            return { batchXs, batchYs, indices: selectedIndices };
        });
    }

    /**
     * Draw 28x28 MNIST image tensor to canvas with optional scaling
     * @param {tf.Tensor} tensor - Image tensor (shape [28,28,1] or [784])
     * @param {HTMLCanvasElement} canvas - Target canvas element
     * @param {number} scale - Scaling factor for better visibility
     */
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        return tf.tidy(() => {
            const ctx = canvas.getContext('2d');
            
            // Ensure tensor is 2D [28,28] and denormalize to 0-255
            const imageData = tensor.reshape([28, 28]).mul(255);
            const data = imageData.dataSync();
            
            // Create ImageData object for the original 28x28 image
            const imgData = new ImageData(28, 28);
            
            // Fill ImageData with grayscale values
            for (let i = 0; i < 784; i++) {
                const val = data[i];
                imgData.data[i * 4] = val;     // R
                imgData.data[i * 4 + 1] = val; // G  
                imgData.data[i * 4 + 2] = val; // B
                imgData.data[i * 4 + 3] = 255; // A (opaque)
            }
            
            // Scale canvas for better visibility
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            ctx.imageSmoothingEnabled = false;
            
            // Draw to temporary canvas then scale up
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imgData, 0, 0);
            
            // Draw scaled image
            ctx.drawImage(tempCanvas, 0, 0, 28 * scale, 28 * scale);
        });
    }

    /**
     * Clean up stored tensor data to prevent memory leaks
     */
    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
