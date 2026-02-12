/**
 * Handwriting Recognition Module
 * 
 * Main entry point for offline, browser-based handwriting recognition.
 * Uses TensorFlow.js with a BiLSTM + CTC model.
 * 
 * Features:
 * - Fully offline (no API calls)
 * - Low latency (< 100ms on desktop)
 * - Supports Nepali (Devanagari), English, digits
 * 
 * @module HandwritingRecognition
 * 
 * Usage:
 *   const recognizer = new HandwritingRecognizer();
 *   await recognizer.init('/models/handwriting');
 *   const strokes = [...captured strokes...];
 *   const result = await recognizer.recognize(strokes);
 *   console.log(result.text);
 */

(function(global) {
    'use strict';

    /**
     * Default configuration
     */
    const DEFAULT_CONFIG = {
        modelPath: '/models/handwriting',
        useBeamSearch: true,
        beamWidth: 10,
        maxSeqLength: 256,
        warmup: true,
        enableProfiling: false
    };

    /**
     * HandwritingRecognizer class
     */
    class HandwritingRecognizer {
        /**
         * @param {Object} config - Configuration options
         */
        constructor(config = {}) {
            this.config = { ...DEFAULT_CONFIG, ...config };
            this.model = null;
            this.vocab = null;
            this.modelConfig = null;
            this.strokeProcessor = null;
            this.ctcDecoder = null;
            this.isInitialized = false;
            this.isLoading = false;
            
            // Performance metrics
            this.metrics = {
                loadTime: 0,
                inferenceTime: 0,
                preprocessTime: 0,
                decodeTime: 0,
                totalCalls: 0
            };
        }

        /**
         * Initialize the recognizer - load model and vocab
         * @param {string} modelPath - Path to model directory
         * @returns {Promise<void>}
         */
        async init(modelPath = null) {
            if (this.isInitialized) return;
            if (this.isLoading) {
                // Wait for existing load to complete
                while (this.isLoading) {
                    await new Promise(r => setTimeout(r, 100));
                }
                return;
            }
            
            this.isLoading = true;
            const startTime = performance.now();
            
            try {
                const basePath = modelPath || this.config.modelPath;
                
                console.log('[HandwritingRecognizer] Initializing...');
                console.log(`  Model path: ${basePath}`);
                
                // Check TensorFlow.js is available
                if (typeof tf === 'undefined') {
                    throw new Error('TensorFlow.js not loaded. Include tf.min.js before handwriting.js');
                }
                
                // Load vocabulary
                console.log('  Loading vocabulary...');
                const vocabResponse = await fetch(`${basePath}/vocab.json`);
                if (!vocabResponse.ok) {
                    throw new Error(`Failed to load vocab.json: ${vocabResponse.status}`);
                }
                this.vocab = await vocabResponse.json();
                console.log(`    Vocabulary size: ${this.vocab.vocab_size}`);
                
                // Load model config
                console.log('  Loading config...');
                const configResponse = await fetch(`${basePath}/config.json`);
                if (!configResponse.ok) {
                    throw new Error(`Failed to load config.json: ${configResponse.status}`);
                }
                this.modelConfig = await configResponse.json();
                console.log(`    Max sequence length: ${this.modelConfig.max_seq_length}`);
                
                // Load TensorFlow.js model
                console.log('  Loading model weights...');
                this.model = await tf.loadLayersModel(`${basePath}/model.json`);
                console.log('    Model loaded successfully');
                
                // Initialize processors
                this.strokeProcessor = new StrokeProcessor({
                    maxSeqLength: this.modelConfig.max_seq_length
                });
                
                this.ctcDecoder = new CTCDecoder(this.vocab, {
                    useBeamSearch: this.config.useBeamSearch,
                    beamWidth: this.config.beamWidth
                });
                
                // Warmup inference
                if (this.config.warmup) {
                    console.log('  Warming up model...');
                    await this._warmup();
                }
                
                this.metrics.loadTime = performance.now() - startTime;
                this.isInitialized = true;
                
                console.log(`[HandwritingRecognizer] Ready! (${this.metrics.loadTime.toFixed(0)}ms)`);
                
            } catch (error) {
                console.error('[HandwritingRecognizer] Initialization failed:', error);
                throw error;
            } finally {
                this.isLoading = false;
            }
        }

        /**
         * Warmup the model with dummy inference
         * @private
         */
        async _warmup() {
            const dummyInput = tf.zeros([1, this.modelConfig.max_seq_length, 3]);
            await this.model.predict(dummyInput).data();
            dummyInput.dispose();
        }

        /**
         * Recognize handwriting from strokes
         * @param {Array} strokes - Array of strokes from StrokeCapture
         * @returns {Promise<Object>} Recognition result
         */
        async recognize(strokes) {
            if (!this.isInitialized) {
                throw new Error('Recognizer not initialized. Call init() first.');
            }
            
            const totalStart = performance.now();
            this.metrics.totalCalls++;
            
            // Preprocess strokes
            const preprocessStart = performance.now();
            const features = this.strokeProcessor.process(strokes);
            const seqLength = this.strokeProcessor.getSequenceLength(strokes);
            this.metrics.preprocessTime = performance.now() - preprocessStart;
            
            if (seqLength === 0) {
                return {
                    text: '',
                    confidence: 0,
                    isEmpty: true,
                    timing: { total: performance.now() - totalStart }
                };
            }
            
            // Run inference
            const inferStart = performance.now();
            
            // Create tensor input (batch size 1)
            const inputTensor = tf.tensor3d(
                Array.from(features),
                [1, this.modelConfig.max_seq_length, 3]
            );
            
            // Get model output
            const outputTensor = this.model.predict(inputTensor);
            const output = await outputTensor.data();
            
            // Clean up tensors
            inputTensor.dispose();
            outputTensor.dispose();
            
            this.metrics.inferenceTime = performance.now() - inferStart;
            
            // Decode output
            const decodeStart = performance.now();
            const result = this.ctcDecoder.decode(output, seqLength);
            this.metrics.decodeTime = performance.now() - decodeStart;
            
            // Add timing info
            result.timing = {
                preprocess: this.metrics.preprocessTime,
                inference: this.metrics.inferenceTime,
                decode: this.metrics.decodeTime,
                total: performance.now() - totalStart
            };
            
            result.seqLength = seqLength;
            result.isEmpty = false;
            
            if (this.config.enableProfiling) {
                console.log(`[Recognize] Total: ${result.timing.total.toFixed(1)}ms`,
                    `(preprocess: ${result.timing.preprocess.toFixed(1)}ms,`,
                    `inference: ${result.timing.inference.toFixed(1)}ms,`,
                    `decode: ${result.timing.decode.toFixed(1)}ms)`);
            }
            
            return result;
        }

        /**
         * Recognize from canvas element directly
         * @param {StrokeCapture} strokeCapture - StrokeCapture instance
         * @returns {Promise<Object>} Recognition result
         */
        async recognizeFromCapture(strokeCapture) {
            const strokes = strokeCapture.getStrokes();
            return this.recognize(strokes);
        }

        /**
         * Get alternative recognition results
         * @param {Array} strokes - Array of strokes
         * @param {number} k - Number of alternatives
         * @returns {Promise<Array>} Array of alternative results
         */
        async getAlternatives(strokes, k = 5) {
            if (!this.isInitialized) {
                throw new Error('Recognizer not initialized. Call init() first.');
            }
            
            const features = this.strokeProcessor.process(strokes);
            const seqLength = this.strokeProcessor.getSequenceLength(strokes);
            
            if (seqLength === 0) {
                return [{ text: '', confidence: 0 }];
            }
            
            const inputTensor = tf.tensor3d(
                Array.from(features),
                [1, this.modelConfig.max_seq_length, 3]
            );
            
            const outputTensor = this.model.predict(inputTensor);
            const output = await outputTensor.data();
            
            inputTensor.dispose();
            outputTensor.dispose();
            
            return this.ctcDecoder.getTopK(output, seqLength, k);
        }

        /**
         * Get performance metrics
         * @returns {Object} Performance metrics
         */
        getMetrics() {
            return { ...this.metrics };
        }

        /**
         * Check if recognizer is ready
         * @returns {boolean}
         */
        isReady() {
            return this.isInitialized;
        }

        /**
         * Dispose resources
         */
        dispose() {
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            this.isInitialized = false;
        }
    }

    /**
     * Convenience function to create and initialize recognizer
     * @param {string} modelPath - Path to model
     * @param {Object} config - Config options
     * @returns {Promise<HandwritingRecognizer>}
     */
    async function createRecognizer(modelPath, config = {}) {
        const recognizer = new HandwritingRecognizer(config);
        await recognizer.init(modelPath);
        return recognizer;
    }

    /**
     * CanvasRecognizer - Combines StrokeCapture and HandwritingRecognizer
     * for easy integration with existing canvas elements
     */
    class CanvasRecognizer {
        /**
         * @param {HTMLCanvasElement} canvas - Canvas element
         * @param {Object} config - Configuration
         */
        constructor(canvas, config = {}) {
            this.canvas = canvas;
            this.config = config;
            this.strokeCapture = new StrokeCapture(canvas);
            this.recognizer = new HandwritingRecognizer(config);
            this.isInitialized = false;
        }

        /**
         * Initialize the recognizer
         * @param {string} modelPath - Model path
         */
        async init(modelPath) {
            await this.recognizer.init(modelPath);
            this.isInitialized = true;
        }

        /**
         * Clear the canvas and strokes
         */
        clear() {
            this.strokeCapture.clear();
        }

        /**
         * Recognize current canvas content
         * @returns {Promise<Object>}
         */
        async recognize() {
            if (!this.isInitialized) {
                throw new Error('Not initialized');
            }
            return this.recognizer.recognizeFromCapture(this.strokeCapture);
        }

        /**
         * Check if canvas has any strokes
         * @returns {boolean}
         */
        hasContent() {
            return this.strokeCapture.hasStrokes();
        }

        /**
         * Get all captured strokes
         * @returns {Array}
         */
        getStrokes() {
            return this.strokeCapture.getStrokes();
        }
    }

    // Export
    global.HandwritingRecognizer = HandwritingRecognizer;
    global.CanvasRecognizer = CanvasRecognizer;
    global.createRecognizer = createRecognizer;

})(typeof window !== 'undefined' ? window : global);
