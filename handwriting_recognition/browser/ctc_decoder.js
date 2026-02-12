/**
 * CTC Decoder Module
 * 
 * Implements Connectionist Temporal Classification (CTC) decoding algorithms
 * for converting model probability outputs to text sequences.
 * 
 * Supports:
 * - Greedy decoding (fast, good for real-time)
 * - Beam search decoding (more accurate, configurable width)
 * 
 * @module CTCDecoder
 */

(function(global) {
    'use strict';

    /**
     * Default decoder configuration
     */
    const DEFAULT_CONFIG = {
        blankIndex: 0,          // CTC blank token index
        beamWidth: 10,          // Beam search width
        blankThreshold: 0.5,    // Threshold for blank detection in greedy
        useBeamSearch: true     // Default to beam search
    };

    /**
     * CTCDecoder class
     */
    class CTCDecoder {
        /**
         * @param {Object} vocab - Vocabulary object with idxToChar mapping
         * @param {Object} config - Decoder configuration
         */
        constructor(vocab, config = {}) {
            this.vocab = vocab;
            this.config = { ...DEFAULT_CONFIG, ...config };
            
            // Build index to character mapping
            this.idxToChar = vocab.idx_to_char || vocab.idxToChar || {};
            this.blankIndex = vocab.blank_index || this.config.blankIndex;
        }

        /**
         * Decode model output probabilities to text
         * @param {Float32Array|Array} probabilities - Model output (seqLen x vocabSize)
         * @param {number} seqLength - Actual sequence length
         * @param {boolean} useBeamSearch - Override beam search setting
         * @returns {Object} Decoded result with text and confidence
         */
        decode(probabilities, seqLength, useBeamSearch = null) {
            const useBeam = useBeamSearch !== null ? useBeamSearch : this.config.useBeamSearch;
            
            if (useBeam) {
                return this.beamSearchDecode(probabilities, seqLength);
            } else {
                return this.greedyDecode(probabilities, seqLength);
            }
        }

        /**
         * Greedy CTC decoding - fast but less accurate
         * @param {Float32Array|Array} probabilities - Model output
         * @param {number} seqLength - Actual sequence length
         * @returns {Object} Decoded result
         */
        greedyDecode(probabilities, seqLength) {
            const vocabSize = this.vocab.vocab_size || Object.keys(this.idxToChar).length;
            const indices = [];
            let totalConfidence = 0;
            let prevIdx = -1;
            
            // Get best class at each timestep
            for (let t = 0; t < seqLength; t++) {
                let maxProb = -Infinity;
                let maxIdx = 0;
                
                for (let c = 0; c < vocabSize; c++) {
                    const prob = probabilities[t * vocabSize + c];
                    if (prob > maxProb) {
                        maxProb = prob;
                        maxIdx = c;
                    }
                }
                
                // CTC collapsing: remove duplicates and blanks
                if (maxIdx !== prevIdx && maxIdx !== this.blankIndex) {
                    indices.push(maxIdx);
                    totalConfidence += maxProb;
                }
                
                prevIdx = maxIdx;
            }
            
            // Convert indices to text
            const text = this._indicesToText(indices);
            const confidence = indices.length > 0 ? totalConfidence / indices.length : 0;
            
            return {
                text: text,
                indices: indices,
                confidence: confidence,
                method: 'greedy'
            };
        }

        /**
         * Beam search CTC decoding - more accurate
         * @param {Float32Array|Array} probabilities - Model output
         * @param {number} seqLength - Actual sequence length
         * @returns {Object} Decoded result
         */
        beamSearchDecode(probabilities, seqLength) {
            const vocabSize = this.vocab.vocab_size || Object.keys(this.idxToChar).length;
            const beamWidth = this.config.beamWidth;
            
            // Initialize beams: {sequence, score, lastChar, logScore}
            let beams = [{
                sequence: [],
                score: 1.0,
                logScore: 0,
                lastChar: -1,
                blankScore: 1.0,
                nonBlankScore: 0
            }];
            
            // Process each timestep
            for (let t = 0; t < seqLength; t++) {
                const newBeams = new Map();
                
                for (const beam of beams) {
                    // For each character in vocabulary
                    for (let c = 0; c < vocabSize; c++) {
                        const prob = probabilities[t * vocabSize + c];
                        
                        if (prob < 1e-6) continue; // Skip very low probabilities
                        
                        if (c === this.blankIndex) {
                            // Blank: keep same sequence
                            const key = beam.sequence.join(',');
                            const newScore = beam.score * prob;
                            
                            if (!newBeams.has(key) || newBeams.get(key).score < newScore) {
                                newBeams.set(key, {
                                    sequence: [...beam.sequence],
                                    score: newScore,
                                    logScore: beam.logScore + Math.log(prob + 1e-10),
                                    lastChar: beam.lastChar,
                                    blankScore: newScore,
                                    nonBlankScore: 0
                                });
                            }
                        } else if (c === beam.lastChar) {
                            // Same character: only extend if previous was blank
                            // This handles repeated characters like 'aa'
                            if (beam.blankScore > 0) {
                                const newSeq = [...beam.sequence, c];
                                const key = newSeq.join(',');
                                const newScore = beam.blankScore * prob;
                                
                                if (!newBeams.has(key) || newBeams.get(key).score < newScore) {
                                    newBeams.set(key, {
                                        sequence: newSeq,
                                        score: newScore,
                                        logScore: beam.logScore + Math.log(prob + 1e-10),
                                        lastChar: c,
                                        blankScore: 0,
                                        nonBlankScore: newScore
                                    });
                                }
                            }
                            
                            // Also allow keeping same sequence
                            const key = beam.sequence.join(',');
                            const newScore = beam.nonBlankScore * prob;
                            
                            if (newBeams.has(key)) {
                                const existing = newBeams.get(key);
                                existing.score += newScore;
                                existing.nonBlankScore += newScore;
                            }
                        } else {
                            // Different character: extend sequence
                            const newSeq = [...beam.sequence, c];
                            const key = newSeq.join(',');
                            const newScore = beam.score * prob;
                            
                            if (!newBeams.has(key) || newBeams.get(key).score < newScore) {
                                newBeams.set(key, {
                                    sequence: newSeq,
                                    score: newScore,
                                    logScore: beam.logScore + Math.log(prob + 1e-10),
                                    lastChar: c,
                                    blankScore: 0,
                                    nonBlankScore: newScore
                                });
                            }
                        }
                    }
                }
                
                // Keep top beams
                beams = Array.from(newBeams.values())
                    .sort((a, b) => b.score - a.score)
                    .slice(0, beamWidth);
                
                if (beams.length === 0) {
                    // Fallback if all beams pruned
                    beams = [{
                        sequence: [],
                        score: 1e-10,
                        logScore: -23,
                        lastChar: -1,
                        blankScore: 1e-10,
                        nonBlankScore: 0
                    }];
                }
            }
            
            // Get best beam
            const bestBeam = beams[0];
            const text = this._indicesToText(bestBeam.sequence);
            
            return {
                text: text,
                indices: bestBeam.sequence,
                confidence: bestBeam.score,
                logScore: bestBeam.logScore,
                method: 'beam_search',
                beamWidth: beamWidth,
                alternatives: beams.slice(1, 5).map(b => ({
                    text: this._indicesToText(b.sequence),
                    confidence: b.score
                }))
            };
        }

        /**
         * Convert index sequence to text
         * @private
         */
        _indicesToText(indices) {
            return indices.map(idx => {
                const char = this.idxToChar[String(idx)] || this.idxToChar[idx] || '';
                return char;
            }).join('');
        }

        /**
         * Get top-k alternative decodings
         * @param {Float32Array|Array} probabilities - Model output
         * @param {number} seqLength - Actual sequence length
         * @param {number} k - Number of alternatives to return
         * @returns {Array} Array of {text, confidence} objects
         */
        getTopK(probabilities, seqLength, k = 5) {
            const result = this.beamSearchDecode(probabilities, seqLength);
            
            const alternatives = [{
                text: result.text,
                confidence: result.confidence
            }];
            
            if (result.alternatives) {
                alternatives.push(...result.alternatives.slice(0, k - 1));
            }
            
            return alternatives;
        }
    }

    /**
     * SimpleCTCDecoder - Simplified greedy decoder for maximum speed
     */
    class SimpleCTCDecoder {
        /**
         * @param {Object} vocab - Vocabulary object
         */
        constructor(vocab) {
            this.idxToChar = vocab.idx_to_char || vocab.idxToChar || {};
            this.blankIndex = vocab.blank_index || 0;
            this.vocabSize = vocab.vocab_size || Object.keys(this.idxToChar).length;
        }

        /**
         * Ultra-fast greedy decode
         * @param {Float32Array} probabilities - Flattened output array
         * @param {number} seqLength - Sequence length
         * @returns {string} Decoded text
         */
        decode(probabilities, seqLength) {
            const chars = [];
            let prevIdx = -1;
            
            for (let t = 0; t < seqLength; t++) {
                const offset = t * this.vocabSize;
                let maxIdx = 0;
                let maxProb = probabilities[offset];
                
                for (let c = 1; c < this.vocabSize; c++) {
                    if (probabilities[offset + c] > maxProb) {
                        maxProb = probabilities[offset + c];
                        maxIdx = c;
                    }
                }
                
                if (maxIdx !== this.blankIndex && maxIdx !== prevIdx) {
                    const char = this.idxToChar[String(maxIdx)] || this.idxToChar[maxIdx];
                    if (char) chars.push(char);
                }
                
                prevIdx = maxIdx;
            }
            
            return chars.join('');
        }
    }

    // Export
    global.CTCDecoder = CTCDecoder;
    global.SimpleCTCDecoder = SimpleCTCDecoder;

})(typeof window !== 'undefined' ? window : global);
