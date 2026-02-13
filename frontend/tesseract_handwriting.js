/**
 * Tesseract.js-based Handwriting Recognition for Sarkari Sarathi
 * 
 * Uses Tesseract OCR with Nepali (nep) and English (eng) language support
 * for recognizing handwritten text from canvas.
 * 
 * Key design: Canvas background is transparent (CSS background is cosmetic only).
 * Strokes are drawn with ctx.strokeStyle='#000' and alpha=255.
 * Preprocessing must:
 *   1. Use ALPHA channel to find strokes (not brightness — transparent = no stroke)
 *   2. Crop to bounding box of actual strokes
 *   3. Upscale and add padding
 *   4. Create clean black-on-white image for Tesseract
 */

(function (global) {
    'use strict';

    let worker = null;
    let isInitialized = false;
    let initPromise = null;

    /**
     * Initialize Tesseract worker with Nepali and English
     */
    async function initTesseract() {
        if (initPromise) return initPromise;

        initPromise = (async () => {
            console.log('[TesseractHandwriting] Initializing...');

            try {
                // Create worker with Nepali language
                worker = await Tesseract.createWorker('nep+eng', 1, {
                    logger: m => {
                        if (m.status === 'recognizing text') {
                            console.log(`[TesseractHandwriting] Progress: ${Math.round(m.progress * 100)}%`);
                        }
                    }
                });

                // Optimal parameters for handwritten Devanagari
                await worker.setParameters({
                    tessedit_pageseg_mode: '6',   // Uniform block of text
                    preserve_interword_spaces: '1',
                    tessedit_char_whitelist: '',   // Allow all characters
                    textord_heavy_nr: '1',         // Assume noisy/handwritten input
                    edges_max_children_per_outline: '40', // Handle complex Devanagari
                });

                isInitialized = true;
                console.log('[TesseractHandwriting] Ready!');

            } catch (error) {
                console.error('[TesseractHandwriting] Init failed:', error);
                throw error;
            }
        })();

        return initPromise;
    }

    /**
     * Find the bounding box of non-transparent pixels (actual strokes)
     * on the original canvas where background is transparent.
     */
    function findStrokeBounds(imageData, width, height) {
        const data = imageData.data;
        let minX = width, minY = height, maxX = 0, maxY = 0;
        let found = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const alpha = data[(y * width + x) * 4 + 3];
                if (alpha > 20) { // Any drawn pixel has alpha > 0
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        if (!found) return null;
        return { minX, minY, maxX, maxY, width: maxX - minX + 1, height: maxY - minY + 1 };
    }

    /**
     * Preprocess canvas for Tesseract OCR.
     * 
     * Strategy:
     * 1. Read raw canvas (transparent bg, black strokes with alpha=255)
     * 2. Find bounding box of strokes using ALPHA channel
     * 3. Crop to bounding box + generous padding
     * 4. Create clean black-on-white binary image
     * 5. Upscale 3x for better Tesseract recognition
     * 6. Apply mild stroke thickening
     */
    function preprocessCanvas(canvas) {
        const origWidth = canvas.width;
        const origHeight = canvas.height;
        const origCtx = canvas.getContext('2d');
        const origData = origCtx.getImageData(0, 0, origWidth, origHeight);

        // Step 1: Find bounding box of drawn strokes
        const bounds = findStrokeBounds(origData, origWidth, origHeight);
        if (!bounds) {
            // No strokes found, return a small white canvas
            const empty = document.createElement('canvas');
            empty.width = 100;
            empty.height = 50;
            const ectx = empty.getContext('2d');
            ectx.fillStyle = 'white';
            ectx.fillRect(0, 0, 100, 50);
            return empty;
        }

        // Step 2: Crop region with padding
        const PADDING = 40;
        const cropX = Math.max(0, bounds.minX - PADDING);
        const cropY = Math.max(0, bounds.minY - PADDING);
        const cropW = Math.min(origWidth - cropX, bounds.width + PADDING * 2);
        const cropH = Math.min(origHeight - cropY, bounds.height + PADDING * 2);

        // Step 3: Create upscaled output canvas
        // Aim for at least ~1800px width for Tesseract (300 DPI equiv)
        const SCALE = Math.max(3, Math.ceil(1800 / cropW));
        const cappedScale = Math.min(SCALE, 5);  // cap at 5x
        const outW = cropW * cappedScale;
        const outH = cropH * cappedScale;

        const outCanvas = document.createElement('canvas');
        outCanvas.width = outW;
        outCanvas.height = outH;
        const outCtx = outCanvas.getContext('2d');

        // Fill with white background
        outCtx.fillStyle = 'white';
        outCtx.fillRect(0, 0, outW, outH);

        // Step 4: Draw the cropped portion of original canvas, scaled up
        // This composites black strokes (alpha=255) onto white background
        outCtx.drawImage(canvas,
            cropX, cropY, cropW, cropH,  // Source rect
            0, 0, outW, outH             // Dest rect (scaled)
        );

        // Step 5: Binarize — convert to pure black-on-white using alpha from source
        // Read the composited result
        const outData = outCtx.getImageData(0, 0, outW, outH);
        const pixels = outData.data;

        // Compute Otsu threshold for adaptive binarization
        const histogram = new Array(256).fill(0);
        for (let i = 0; i < pixels.length; i += 4) {
            const brightness = Math.round((pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3);
            histogram[brightness]++;
        }
        const totalPixels = pixels.length / 4;
        let sumAll = 0;
        for (let t = 0; t < 256; t++) sumAll += t * histogram[t];
        let sumBg = 0, wBg = 0, maxVariance = 0, bestThreshold = 128;
        for (let t = 0; t < 256; t++) {
            wBg += histogram[t];
            if (wBg === 0) continue;
            const wFg = totalPixels - wBg;
            if (wFg === 0) break;
            sumBg += t * histogram[t];
            const meanBg = sumBg / wBg;
            const meanFg = (sumAll - sumBg) / wFg;
            const variance = wBg * wFg * (meanBg - meanFg) * (meanBg - meanFg);
            if (variance > maxVariance) {
                maxVariance = variance;
                bestThreshold = t;
            }
        }
        // Pad threshold slightly to preserve thin strokes
        const binThreshold = Math.min(bestThreshold + 20, 220);

        for (let i = 0; i < pixels.length; i += 4) {
            const brightness = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;

            if (brightness < binThreshold) {
                pixels[i] = 0;
                pixels[i + 1] = 0;
                pixels[i + 2] = 0;
                pixels[i + 3] = 255;
            } else {
                pixels[i] = 255;
                pixels[i + 1] = 255;
                pixels[i + 2] = 255;
                pixels[i + 3] = 255;
            }
        }

        // Step 6: Two-pass dilation — thicken strokes for handwriting
        // This helps Tesseract with thin hand-drawn strokes
        const dilated = new Uint8ClampedArray(pixels.length);
        dilated.set(pixels); // copy

        // Two passes of 4-connected dilation
        for (let pass = 0; pass < 2; pass++) {
            const src = pass === 0 ? pixels : new Uint8ClampedArray(dilated);
            for (let y = 1; y < outH - 1; y++) {
                for (let x = 1; x < outW - 1; x++) {
                    const idx = (y * outW + x) * 4;
                    if (src[idx] === 0) { dilated[idx] = 0; dilated[idx+1] = 0; dilated[idx+2] = 0; dilated[idx+3] = 255; continue; }

                    const up    = ((y - 1) * outW + x) * 4;
                    const down  = ((y + 1) * outW + x) * 4;
                    const left  = (y * outW + (x - 1)) * 4;
                    const right = (y * outW + (x + 1)) * 4;

                    if (src[up] === 0 || src[down] === 0 ||
                        src[left] === 0 || src[right] === 0) {
                        dilated[idx] = 0;
                        dilated[idx + 1] = 0;
                        dilated[idx + 2] = 0;
                        dilated[idx + 3] = 255;
                    }
                }
            }
        }

        const finalImageData = new ImageData(dilated, outW, outH);
        outCtx.putImageData(finalImageData, 0, 0);

        console.log(`[TesseractHandwriting] Preprocessed: crop(${cropX},${cropY} ${cropW}x${cropH}) → ${outW}x${outH}`);
        return outCanvas;
    }

    /**
     * Recognize text from canvas
     */
    async function recognizeFromCanvas(canvas) {
        if (!isInitialized) {
            await initTesseract();
        }

        console.log('[TesseractHandwriting] Starting recognition...');
        const startTime = performance.now();

        // Preprocess the canvas
        const processedCanvas = preprocessCanvas(canvas);

        // Convert to blob for Tesseract
        const blob = await new Promise(resolve => {
            processedCanvas.toBlob(resolve, 'image/png');
        });

        try {
            // Try multiple PSM modes and pick the best result
            let bestText = '';
            let bestConf = 0;

            // PSM 6 = Uniform block of text (default, already set)
            const result6 = await worker.recognize(blob);
            const text6 = result6.data.text.trim();
            const conf6 = result6.data.confidence;
            console.log(`[TesseractHandwriting] PSM6: "${text6}" (conf: ${conf6})`);

            if (conf6 > bestConf && text6.length > 0) {
                bestText = text6;
                bestConf = conf6;
            }

            // Always try PSM 7 (single text line) — often better for single-field input
            try {
                await worker.setParameters({ tessedit_pageseg_mode: '7' });
                const result7 = await worker.recognize(blob);
                const text7 = result7.data.text.trim();
                const conf7 = result7.data.confidence;
                console.log(`[TesseractHandwriting] PSM7: "${text7}" (conf: ${conf7})`);

                if (conf7 > bestConf && text7.length > 0) {
                    bestText = text7;
                    bestConf = conf7;
                }
            } catch(e) {
                console.warn('[TesseractHandwriting] PSM7 attempt failed:', e);
            }

            // Also try PSM 13 (raw line) for very short text like names
            try {
                await worker.setParameters({ tessedit_pageseg_mode: '13' });
                const result13 = await worker.recognize(blob);
                const text13 = result13.data.text.trim();
                const conf13 = result13.data.confidence;
                console.log(`[TesseractHandwriting] PSM13: "${text13}" (conf: ${conf13})`);

                if (conf13 > bestConf && text13.length > 0) {
                    bestText = text13;
                    bestConf = conf13;
                }
            } catch(e) {
                console.warn('[TesseractHandwriting] PSM13 attempt failed:', e);
            }

            // Reset back to PSM 6 for next call
            await worker.setParameters({ tessedit_pageseg_mode: '6' });

            // Filter out garbage results (very short, only special chars, etc.)
            bestText = cleanOCRResult(bestText);

            const duration = Math.round(performance.now() - startTime);
            console.log(`[TesseractHandwriting] Final: "${bestText}" (conf: ${bestConf}, ${duration}ms)`);

            return {
                success: bestText.length > 0,
                text: bestText,
                confidence: bestConf,
                duration: duration
            };

        } catch (error) {
            console.error('[TesseractHandwriting] Recognition error:', error);
            return {
                success: false,
                text: '',
                error: error.message
            };
        }
    }

    /**
     * Clean OCR output — remove garbage characters common in handwriting misreads
     */
    function cleanOCRResult(text) {
        if (!text) return '';
        
        // Remove control characters and excessive whitespace
        text = text.replace(/[\x00-\x1F\x7F]/g, '').trim();
        
        // Remove lines that are only punctuation/symbols (noise)
        const lines = text.split('\n');
        const cleanLines = lines.filter(line => {
            const trimmed = line.trim();
            if (trimmed.length === 0) return false;
            // Keep lines with at least one alphanumeric or Devanagari character
            return /[\u0900-\u097F\u0980-\u09FFa-zA-Z0-9]/.test(trimmed);
        });
        
        text = cleanLines.join(' ').trim();
        
        // Collapse multiple spaces
        text = text.replace(/\s+/g, ' ');
        
        return text;
    }

    /**
     * Check if Tesseract is available
     */
    function isAvailable() {
        return typeof Tesseract !== 'undefined';
    }

    /**
     * Check if initialized
     */
    function isReady() {
        return isInitialized;
    }

    // Export to global
    global.TesseractHandwriting = {
        init: initTesseract,
        recognize: recognizeFromCanvas,
        preprocess: preprocessCanvas,  // Expose for debugging
        isAvailable: isAvailable,
        isReady: isReady
    };

})(window);
