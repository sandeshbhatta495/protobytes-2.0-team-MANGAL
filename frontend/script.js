let currentStep = 1;
let selectedDocument = null;
let selectedInputMethod = 'text';
let formData = {};


// Field-centric state store
const fieldStates = {};

// Modal canvas state
let activeCanvasFieldId = null;
let isModalDrawing = false;
let modalLastX = 0;
let modalLastY = 0;

function getApiBase() {
    var o = window.location.origin;
    var path = window.location.pathname || '';
    if (path.indexOf('/app/') === 0 || path === '/app' || path === '/') {
        return o;
    }
    var host = window.location.hostname || 'localhost';
    return (o && o.indexOf('https') === 0 ? 'https' : 'http') + '://' + host + ':8000';
}

const API_BASE = getApiBase();

// =====================================================
//  FIELD STATE CLASS
//  Central state for each form field. All input modes
//  (typing, voice, writing) funnel through setValue().
// =====================================================
class FieldState {
    constructor(fieldId, element, label) {
        this.fieldId = fieldId;
        this.element = element;
        this.label = label || fieldId;
        this.value = '';
        this.activeInputMode = 'typing'; // typing | voice | writing
        // Per-field voice recording state
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        // Transliteration buffer (current English word being composed)
        this.translitBuffer = '';
    }

/** Set the field value from any source and sync to DOM */
    setValue(newValue, source) {
        this.value = newValue;
        if (this.element) {
            this.element.value = newValue;
            this.element.dispatchEvent(new Event('change', { bubbles: true }));
            this.element.dispatchEvent(new Event('input', { bubbles: true }));
        }
        console.log(`[FieldState] ${this.fieldId} = "${newValue}" (source: ${source || 'unknown'})`);
    }
   /** Append text (e.g. from voice transcription) */
    appendValue(text, source) {
        const sep = this.value ? ' ' : '';
        this.setValue(this.value + sep + text, source);
    }

    /** Read current value from DOM (sync back) */
    getValue() {
        if (this.element) this.value = this.element.value;
        return this.value;
    }
}

function getOrCreateFieldState(fieldId, element, label) {
    if (!fieldStates[fieldId]) {
        fieldStates[fieldId] = new FieldState(fieldId, element, label);
    } else if (element) {
        fieldStates[fieldId].element = element;
        if (label) fieldStates[fieldId].label = label;
    }
    return fieldStates[fieldId];
}

// =====================================================
//  CLIENT-SIDE TRANSLITERATION (English → Nepali)
// =====================================================
const TRANSLIT_MULTI = {
    'shri': 'श्री', 'shr': 'श्र', 'ksh': 'क्ष', 'tra': 'त्र', 'gya': 'ज्ञ',
    'chh': 'छ', 'thh': 'ठ', 'dhh': 'ढ', 'shh': 'ष',
    'kha': 'खा', 'gha': 'घा', 'cha': 'चा', 'chha': 'छा',
    'jha': 'झा', 'tha': 'था', 'dha': 'धा', 'pha': 'फा',
    'bha': 'भा', 'sha': 'शा',
    'kh': 'ख', 'gh': 'घ', 'ng': 'ङ',
    'ch': 'च', 'jh': 'झ', 'ny': 'ञ',
    'th': 'थ', 'dh': 'ध', 'ph': 'फ',
    'bh': 'भ', 'sh': 'श',
    'aa': 'ा', 'ee': 'ी', 'oo': 'ू', 'ai': 'ै', 'au': 'ौ',
    'ou': 'ौ', 'ei': 'ै'
};
const TRANSLIT_VOWEL_STANDALONE = { 'a': 'अ', 'i': 'इ', 'u': 'उ', 'e': 'ए', 'o': 'ओ' };
const TRANSLIT_VOWEL_MATRA = { 'a': '', 'i': 'ि', 'u': 'ु', 'e': 'े', 'o': 'ो' };
const TRANSLIT_CONSONANT = {
    'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 't': 'त',
    'd': 'द', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म',
    'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
    's': 'स', 'h': 'ह', 'f': 'फ', 'z': 'ज़', 'x': 'क्स', 'q': 'क'
};
const DEVANAGARI_DIGITS = '०१२३४५६७८९';
const TRANSLIT_WORDS = {
    'name': 'नाम', 'first': 'पहिलो', 'last': 'थर', 'ram': 'राम',
    'sita': 'सिता', 'hari': 'हरि', 'kumar': 'कुमार', 'shrestha': 'श्रेष्ठ',
    'sharma': 'शर्मा', 'thapa': 'थापा', 'tamang': 'तामाङ', 'gurung': 'गुरुङ',
    'magar': 'मगर', 'rai': 'राई', 'limbu': 'लिम्बु', 'nepal': 'नेपाल',
    'kathmandu': 'काठमाडौं', 'pokhara': 'पोखरा', 'lalitpur': 'ललितपुर',
    'bhaktapur': 'भक्तपुर', 'biratnagar': 'बिराटनगर',
    'male': 'पुरुष', 'female': 'महिला', 'other': 'अन्य',
    'married': 'विवाहित', 'unmarried': 'अविवाहित', 'single': 'एकल',
    'hindu': 'हिन्दू', 'buddhist': 'बौद्ध', 'muslim': 'मुस्लिम', 'christian': 'ईसाई',
    'nepali': 'नेपाली', 'father': 'बुबा', 'mother': 'आमा',
    'son': 'छोरा', 'daughter': 'छोरी', 'husband': 'पति', 'wife': 'पत्नी',
    'grandfather': 'हजुरबुबा', 'grandmother': 'हजुरआमा',
    'bahadur': 'बहादुर', 'prasad': 'प्रसाद', 'devi': 'देवी', 'maya': 'माया',
    'laxmi': 'लक्ष्मी', 'krishna': 'कृष्ण', 'shiva': 'शिव', 'ganesh': 'गणेश',
    'bir': 'बिर', 'dal': 'दल', 'jit': 'जित'
};

/** Transliterate a single English word/phrase to Nepali Devanagari */
function clientTransliterate(text) {
    if (!text) return '';
    const lower = text.toLowerCase().trim();
    if (TRANSLIT_WORDS[lower]) return TRANSLIT_WORDS[lower];

    let result = [];
    let i = 0;
    let afterConsonant = false;

    while (i < text.length) {
        const char = text[i].toLowerCase();
        let matched = false;

        if (char === ' ') { result.push(' '); afterConsonant = false; i++; continue; }
        if ('.,;:!?()[]{}"\'-/\\@#$%^&*+=<>|~`'.includes(char)) {
            result.push(char); afterConsonant = false; i++; continue;
        }
        if (char >= '0' && char <= '9') {
            result.push(DEVANAGARI_DIGITS[parseInt(char)]); afterConsonant = false; i++; continue;
        }
        // Already Devanagari? Pass through
        const code = char.charCodeAt(0);
        if (code >= 0x0900 && code <= 0x097F) { result.push(text[i]); afterConsonant = false; i++; continue; }

        // Multi-char matches (longest first)
        for (let len = 4; len >= 2; len--) {
            const substr = text.substring(i, i + len).toLowerCase();
            if (TRANSLIT_MULTI[substr]) {
                result.push(TRANSLIT_MULTI[substr]);
                afterConsonant = !'aeiou'.includes(substr[substr.length - 1]);
                i += len; matched = true; break;
            }
        }
        if (matched) continue;

        if (TRANSLIT_CONSONANT[char]) {
            if (afterConsonant) result.push('्');
            result.push(TRANSLIT_CONSONANT[char]);
            afterConsonant = true; i++;
        } else if (TRANSLIT_VOWEL_STANDALONE[char]) {
            if (afterConsonant) result.push(TRANSLIT_VOWEL_MATRA[char]);
            else result.push(TRANSLIT_VOWEL_STANDALONE[char]);
            afterConsonant = false; i++;
        } else {
            result.push(text[i]); afterConsonant = false; i++;
        }
    }
    return result.join('');
}


