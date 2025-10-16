class DataLoader {
    constructor() {
        this.stocksData = null;
        this.normalizedData = null;
        this.symbols = null;
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
    }

    // Parse CSV file from file input
    async parseCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    const lines = csv.split('\n').filter(line => line.trim());
                    const headers = lines[0].split(',').map(h => h.trim());
                    
                    const data = [];
                    for (let i = 1; i < lines.length; i++) {
                        const values = lines[i].split(',').map(v => v.trim());
                        const row = {};
                        headers.forEach((header, index) => {
                            row[header] = values[index];
                        });
                        data.push(row);
                    }
                    
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    // Pivot data to align dates and symbols
    pivotData(data) {
        const symbols = [...new Set(data.map(row => row.Symbol))];
        const dates = [...new Set(data.map(row => row.Date))].sort();
        
        const pivoted = {};
        symbols.forEach(symbol => {
            pivoted[symbol] = {};
            dates.forEach(date => {
                const row = data.find(r => r.Date === date && r.Symbol === symbol);
                if (row) {
                    pivoted[symbol][date] = {
                        Open: parseFloat(row.Open),
                        Close: parseFloat(row.Close)
                    };
                }
            });
        });
        
        return { pivoted, symbols, dates };
    }

    // Normalize data using MinMax scaling per stock
    normalizeData(pivotedData) {
        const normalized = {};
        const minMax = {};
        
        Object.keys(pivotedData).forEach(symbol => {
            const values = Object.values(pivotedData[symbol]);
            const opens = values.map(v => v.Open);
            const closes = values.map(v => v.Close);
            
            minMax[symbol] = {
                Open: { min: Math.min(...opens), max: Math.max(...opens) },
                Close: { min: Math.min(...closes), max: Math.max(...closes) }
            };
            
            normalized[symbol] = {};
            Object.keys(pivotedData[symbol]).forEach(date => {
                const original = pivotedData[symbol][date];
                normalized[symbol][date] = {
                    Open: (original.Open - minMax[symbol].Open.min) / 
                          (minMax[symbol].Open.max - minMax[symbol].Open.min),
                    Close: (original.Close - minMax[symbol].Close.min) / 
                           (minMax[symbol].Close.max - minMax[symbol].Close.min)
                };
            });
        });
        
        return { normalized, minMax };
    }

    // Prepare sliding window samples
    prepareSamples(normalizedData, symbols, dates, sequenceLength = 12, predictionDays = 3) {
        const samples = [];
        const targets = [];
        
        const featureDates = dates.slice(0, dates.length - predictionDays);
        
        for (let i = sequenceLength; i < featureDates.length; i++) {
            const currentDate = featureDates[i];
            const sequenceStart = i - sequenceLength;
            
            // Input: 12 days of [Open, Close] for all 10 stocks
            const input = [];
            for (let j = sequenceStart; j < i; j++) {
                const date = featureDates[j];
                const features = [];
                symbols.forEach(symbol => {
                    if (normalizedData[symbol] && normalizedData[symbol][date]) {
                        features.push(normalizedData[symbol][date].Open);
                        features.push(normalizedData[symbol][date].Close);
                    } else {
                        features.push(0, 0); // Handle missing data
                    }
                });
                input.push(features);
            }
            
            // Output: 3-day-ahead binary classification for each stock
            const target = [];
            const currentClosePrices = {};
            symbols.forEach(symbol => {
                if (normalizedData[symbol] && normalizedData[symbol][currentDate]) {
                    currentClosePrices[symbol] = normalizedData[symbol][currentDate].Close;
                } else {
                    currentClosePrices[symbol] = 0;
                }
            });
            
            for (let offset = 1; offset <= predictionDays; offset++) {
                const futureDateIndex = dates.indexOf(currentDate) + offset;
                if (futureDateIndex < dates.length) {
                    const futureDate = dates[futureDateIndex];
                    symbols.forEach(symbol => {
                        if (normalizedData[symbol] && normalizedData[symbol][futureDate]) {
                            const futureClose = normalizedData[symbol][futureDate].Close;
                            const label = futureClose > currentClosePrices[symbol] ? 1 : 0;
                            target.push(label);
                        } else {
                            target.push(0); // Handle missing data
                        }
                    });
                } else {
                    // Padding if we don't have enough future data
                    symbols.forEach(() => target.push(0));
                }
            }
            
            samples.push(input);
            targets.push(target);
        }
        
        return { samples, targets };
    }

    // Split data into train/test sets
    splitData(samples, targets, splitRatio = 0.8) {
        const splitIndex = Math.floor(samples.length * splitRatio);
        
        const X_train = samples.slice(0, splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const X_test = samples.slice(splitIndex);
        const y_test = targets.slice(splitIndex);
        
        return {
            X_train: tf.tensor3d(X_train),
            y_train: tf.tensor2d(y_train),
            X_test: tf.tensor3d(X_test),
            y_test: tf.tensor2d(y_test)
        };
    }

    // Main loading function
    async loadData(file) {
        try {
            // Parse CSV
            const rawData = await this.parseCSV(file);
            
            // Pivot data
            const { pivoted, symbols, dates } = this.pivotData(rawData);
            
            // Normalize data
            const { normalized } = this.normalizeData(pivoted);
            
            // Prepare samples
            const { samples, targets } = this.prepareSamples(normalized, symbols, dates);
            
            // Split data
            const { X_train, y_train, X_test, y_test } = this.splitData(samples, targets);
            
            this.stocksData = rawData;
            this.normalizedData = normalized;
            this.symbols = symbols;
            this.X_train = X_train;
            this.y_train = y_train;
            this.X_test = X_test;
            this.y_test = y_test;
            
            return {
                symbols,
                X_train, y_train, X_test, y_test,
                sampleCount: samples.length,
                trainSize: X_train.shape[0],
                testSize: X_test.shape[0]
            };
            
        } catch (error) {
            console.error('Error loading data:', error);
            throw error;
        }
    }

    // Clean up tensors
    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}

export default DataLoader;