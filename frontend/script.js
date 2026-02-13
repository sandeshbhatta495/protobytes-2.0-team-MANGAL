// =====================================================
//  SARKARI-SARATHI — FIELD-CENTRIC INPUT ARCHITECTURE
//  Every form field has a single internal state.
//  All input methods (keyboard, voice, handwriting)
//  write to that same state → always Nepali output.
// =====================================================

// ===== GLOBAL VARIABLES =====
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

// CNN alternatives state
let pendingAlternatives = [];
let selectedWordIndex = 0;
let lastRecognitionResult = null;

// ===== API BASE URL =====
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
    // Conjuncts and special combinations
    'shri': 'श्री', 'shree': 'श्री',
    'kshya': 'क्ष्य', 'ksha': 'क्षा', 'ksh': 'क्ष',
    'gya': 'ज्ञ', 'dnya': 'ज्ञ', 'gnya': 'ज्ञ',
    'tra': 'त्र', 'tri': 'त्रि', 'tru': 'त्रु',
    'dra': 'द्र', 'dri': 'द्रि',
    'pra': 'प्र', 'pri': 'प्रि',
    'bra': 'ब्र', 'bri': 'ब्रि',
    'shr': 'श्र', 'shra': 'श्रा',
    'ntr': 'न्त्र', 'ndr': 'न्द्र',
    'str': 'स्त्र', 'sta': 'स्ता',
    'sth': 'स्थ', 'stha': 'स्था',
    'sna': 'स्ना', 'swa': 'स्वा', 'sw': 'स्व',
    'tth': 'त्थ', 'ddh': 'द्ध',
    'nch': 'न्च', 'nj': 'न्ज', 'nd': 'न्द', 'nt': 'न्त',
    'mp': 'म्प', 'mb': 'म्ब',
    'ng': 'ङ', 'nk': 'ङ्क',
    'rya': 'र्य', 'ryu': 'र्यु',
    // Aspirated consonants with vowels
    'chha': 'छा', 'chhi': 'छि', 'chhu': 'छु', 'chhe': 'छे', 'chho': 'छो',
    'chh': 'छ',
    'kha': 'खा', 'khi': 'खि', 'khu': 'खु', 'khe': 'खे', 'kho': 'खो',
    'kh': 'ख',
    'gha': 'घा', 'ghi': 'घि', 'ghu': 'घु', 'ghe': 'घे', 'gho': 'घो',
    'gh': 'घ',
    'cha': 'चा', 'chi': 'चि', 'chu': 'चु', 'che': 'चे', 'cho': 'चो',
    'ch': 'च',
    'jha': 'झा', 'jhi': 'झि', 'jhu': 'झु', 'jhe': 'झे', 'jho': 'झो',
    'jh': 'झ',
    'tha': 'था', 'thi': 'थि', 'thu': 'थु', 'the': 'थे', 'tho': 'थो',
    'th': 'थ',
    'dha': 'धा', 'dhi': 'धि', 'dhu': 'धु', 'dhe': 'धे', 'dho': 'धो',
    'dh': 'ध',
    'pha': 'फा', 'phi': 'फि', 'phu': 'फु', 'phe': 'फे', 'pho': 'फो',
    'ph': 'फ',
    'bha': 'भा', 'bhi': 'भि', 'bhu': 'भु', 'bhe': 'भे', 'bho': 'भो',
    'bh': 'भ',
    'sha': 'शा', 'shi': 'शि', 'shu': 'शु', 'she': 'शे', 'sho': 'शो',
    'sh': 'श',
    'ny': 'ञ',
    // Vowel combinations
    'aa': 'ा', 'ee': 'ी', 'ii': 'ी', 'oo': 'ू', 'uu': 'ू',
    'ai': 'ै', 'au': 'ौ', 'ou': 'ौ', 'ei': 'ै',
    // Retroflex
    'tt': 'ट', 'tth': 'ठ', 'dd': 'ड', 'ddh': 'ढ', 'nn': 'ण'
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
    // Personal names and relations
    'name': 'नाम', 'first': 'पहिलो', 'last': 'थर', 'ram': 'राम',
    'sita': 'सीता', 'hari': 'हरि', 'kumar': 'कुमार', 'shrestha': 'श्रेष्ठ',
    'sharma': 'शर्मा', 'thapa': 'थापा', 'tamang': 'तामाङ', 'gurung': 'गुरुङ',
    'magar': 'मगर', 'rai': 'राई', 'limbu': 'लिम्बु', 'newar': 'नेवार',
    'bahadur': 'बहादुर', 'prasad': 'प्रसाद', 'devi': 'देवी', 'maya': 'माया',
    'laxmi': 'लक्ष्मी', 'krishna': 'कृष्ण', 'shiva': 'शिव', 'ganesh': 'गणेश',
    'bir': 'बिर', 'dal': 'दल', 'jit': 'जित',
    // Family relations
    'father': 'बुबा', 'mother': 'आमा', 'son': 'छोरा', 'daughter': 'छोरी',
    'husband': 'पति', 'wife': 'पत्नी', 
    'grandfather': 'हजुरबुबा', 'grandmother': 'हजुरआमा',
    'brother': 'दाजु', 'sister': 'दिदी',
    // Places
    'nepal': 'नेपाल', 'kathmandu': 'काठमाडौं', 'pokhara': 'पोखरा', 
    'lalitpur': 'ललितपुर', 'bhaktapur': 'भक्तपुर', 'biratnagar': 'बिराटनगर',
    // Gender and status
    'male': 'पुरुष', 'female': 'महिला', 'other': 'अन्य',
    'married': 'विवाहित', 'unmarried': 'अविवाहित', 'single': 'एकल',
    // Religion
    'hindu': 'हिन्दू', 'buddhist': 'बौद्ध', 'muslim': 'मुस्लिम', 'christian': 'ईसाई',
    'nepali': 'नेपाली',
    // Administrative terms
    'province': 'प्रदेश', 'district': 'जिल्ला', 'ward': 'वडा',
    'municipality': 'नगरपालिका', 'village': 'गाउँ', 'address': 'ठेगाना',
    'date': 'मिति', 'year': 'वर्ष', 'month': 'महिना', 'day': 'दिन',
    // Document types
    'birth': 'जन्म', 'death': 'मृत्यु', 'marriage': 'विवाह', 'divorce': 'सम्बन्ध विच्छेद',
    'application': 'निवेदन', 'certificate': 'प्रमाणपत्र', 'registration': 'दर्ता'
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

// =====================================================
//  NEPALI TEXT VALIDATION
// =====================================================
function isDevanagariChar(ch) {
    const c = ch.charCodeAt(0);
    return (c >= 0x0900 && c <= 0x097F);
}

function isNepaliText(text) {
    if (!text || !text.trim()) return true;
    const cleaned = text.replace(/[\s\d.,;:!?()\-\/'"।०-९]+/g, '');
    if (!cleaned) return true;
    let nepaliCount = 0;
    for (const ch of cleaned) { if (isDevanagariChar(ch)) nepaliCount++; }
    return (nepaliCount / cleaned.length) >= 0.5;
}

// Helper to check if a field is for English input (should NOT be transliterated)
function isEnglishField(fieldId) {
    return fieldId && (fieldId.endsWith('_en') || fieldId.endsWith('_english') || fieldId.toLowerCase().includes('english'));
}

function validateAllFieldsNepali() {
    const issues = [];
    const form = document.getElementById('documentForm');
    if (!form) return issues;
    form.querySelectorAll('input[type="text"], textarea').forEach(input => {
        const id = input.id || '';
        const name = input.name || '';
        // Skip fields that are explicitly marked as English variants
        if (isEnglishField(id) || isEnglishField(name)) {
            return;
        }
        const val = input.value.trim();
        if (val && !isNepaliText(val)) {
            const state = fieldStates[id];
            issues.push({ fieldId: id, label: state ? state.label : id, value: val });
        }
    });
    return issues;
}

// =====================================================
//  TOAST NOTIFICATIONS (replaces alert)
// =====================================================
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function () {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(function () { toast.remove(); }, 300);
    }, duration);
}

function showError(message) { showToast(message, 'error', 5000); }
function showSuccess(message) { showToast(message, 'success', 3000); }

// =====================================================
//  APP INITIALIZATION
// =====================================================
document.addEventListener('DOMContentLoaded', function () {
    try {
        loadInitialData();
        setupEventListeners();
        setupModalCanvas();
        
        // Initialize Tesseract.js for handwriting recognition (async, non-blocking)
        if (typeof TesseractHandwriting !== 'undefined' && TesseractHandwriting.isAvailable()) {
            console.log('[Init] Pre-loading Tesseract.js OCR engine...');
            TesseractHandwriting.init().then(function() {
                console.log('[Init] Tesseract.js ready for handwriting recognition');
            }).catch(function(err) {
                console.warn('[Init] Tesseract.js init failed:', err);
            });
        }
    } catch (err) {
        console.error('App init error:', err);
    }
});

function setupEventListeners() {
    var form = document.getElementById('documentForm');
    if (form) form.addEventListener('submit', handleFormSubmit);
}

async function loadDocumentTypes() {
    try {
        const response = await fetch(API_BASE + '/document-types');
        const data = await response.json();
        console.log('Available document types:', data);
    } catch (error) {
        console.error('Error loading document types:', error);
    }
}

// Map step number to content div id (3-step flow: Doc Select → Form → Preview)
const STEP_CONTENT_IDS = { 1: 'documentSelection', 2: 'formInput', 3: 'previewDownload' };

function goToStep(step) {
    document.querySelectorAll('.step-content').forEach(function (c) { c.classList.add('hidden'); });
    document.querySelectorAll('.step-indicator').forEach(function (ind) {
        ind.classList.remove('active'); ind.classList.add('bg-gray-200');
    });
    var contentId = STEP_CONTENT_IDS[step];
    if (contentId) { var el = document.getElementById(contentId); if (el) el.classList.remove('hidden'); }
    var stepEl = document.getElementById('step' + step);
    if (stepEl) { stepEl.classList.remove('bg-gray-200'); stepEl.classList.add('active'); }
    currentStep = step;
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/** Safe navigation: only go to step if prerequisite is met */
function goToStepSafe(step) {
    if (step === 2 && !selectedDocument) {
        showError('कृपया पहिले दस्तावेज छनोट गर्नुहोस्।');
        return;
    }
    if (step === 3 && !window.formData) {
        showError('कृपया पहिले फारम भर्नुहोस्।');
        return;
    }
    goToStep(step);
}

/** Go back from preview to form — data is preserved */
function goBackToForm() {
    goToStep(2);
    showToast('फारम सम्पादन गर्न सक्नुहुन्छ — डाटा सुरक्षित छ।', 'info', 2500);
}

function selectDocument(documentType) {
    selectedDocument = documentType;
    console.log('Selected document:', documentType);
    showLoading();
    loadDocumentTemplate(documentType)
        .then(function () {
            hideLoading();
            // Go directly to form (step 2) — no input method selection needed
            goToStep(2);
            showToast('प्रत्येक फिल्डमा किबोर्ड, माइक वा पेन प्रयोग गर्न सकिन्छ', 'info', 3000);
        })
        .catch(function (error) {
            hideLoading();
            var msg = (error && error.message) ? error.message : 'दस्तावेज टेम्प्लेट लोड गर्न सकेन।';
            showError(msg + ' सर्भर: ' + API_BASE);
        });
}

async function loadDocumentTemplate(documentType) {
    var url = API_BASE + '/template/' + documentType;
    try {
        console.log('Fetching template:', url);
        var response = await fetch(url);
        if (!response.ok) {
            var msg = response.status === 404
                ? 'Template not found. Make sure the server is running.'
                : 'Server error: ' + response.status;
            throw new Error(msg);
        }
        var template = await response.json();
        console.log('Template loaded:', template);
        if (template.form_fields && Array.isArray(template.form_fields)) {
            await ensureLocationData();
            generateFormFields(template.form_fields);
        } else {
            throw new Error('Invalid template format: missing form_fields');
        }
    } catch (error) {
        console.error('Error loading template:', error);
        throw error;
    }
}

// Global location data
var locationData = null;
var locationDataPromise = null;

// =====================================================
//  LOCATION DATA
// =====================================================
var NEPAL_LOCATIONS_FALLBACK = {
    country: {
        provinces: [
            { province_id: 1, province_name: 'कोशी प्रदेश', districts: [{ district_name: 'भोजपुर' }, { district_name: 'धनकुटा' }, { district_name: 'इलाम' }, { district_name: 'झापा' }, { district_name: 'खोटाङ' }, { district_name: 'मोरङ' }, { district_name: 'ओखलढुंगा' }, { district_name: 'पाँचथर' }, { district_name: 'संखुवासभा' }, { district_name: 'सोलुखुम्बु' }, { district_name: 'सुनसरी' }, { district_name: 'ताप्लेजुङ' }, { district_name: 'तेह्रथुम' }, { district_name: 'उदयपुर' }] },
            { province_id: 2, province_name: 'मधेश प्रदेश', districts: [{ district_name: 'बारा' }, { district_name: 'धनुषा' }, { district_name: 'महोत्तरी' }, { district_name: 'पर्सा' }, { district_name: 'रौतहट' }, { district_name: 'सप्तरी' }, { district_name: 'सर्लाही' }, { district_name: 'सिरहा' }] },
            { province_id: 3, province_name: 'बागमती प्रदेश', districts: [{ district_name: 'भक्तपुर' }, { district_name: 'चितवन' }, { district_name: 'धादिङ' }, { district_name: 'दोलखा' }, { district_name: 'काठमाडौं' }, { district_name: 'काभ्रेपलाञ्चोक' }, { district_name: 'ललितपुर' }, { district_name: 'मकवानपुर' }, { district_name: 'नुवाकोट' }, { district_name: 'रामेछाप' }, { district_name: 'रसुवा' }, { district_name: 'सिन्धुली' }, { district_name: 'सिन्धुपाल्चोक' }] },
            { province_id: 4, province_name: 'गण्डकी प्रदेश', districts: [{ district_name: 'बागलुङ' }, { district_name: 'गोरखा' }, { district_name: 'कास्की' }, { district_name: 'लमजुङ' }, { district_name: 'मनाङ' }, { district_name: 'मुस्ताङ' }, { district_name: 'म्याग्दी' }, { district_name: 'नवलपुर' }, { district_name: 'पर्वत' }, { district_name: 'स्याङ्जा' }, { district_name: 'तनहुँ' }] },
            { province_id: 5, province_name: 'लुम्बिनी प्रदेश', districts: [{ district_name: 'अर्घाखाँची' }, { district_name: 'बाँके' }, { district_name: 'बर्दिया' }, { district_name: 'दाङ' }, { district_name: 'गुल्मी' }, { district_name: 'कपिलवस्तु' }, { district_name: 'पाल्पा' }, { district_name: 'परासी' }, { district_name: 'प्यूठान' }, { district_name: 'रोल्पा' }, { district_name: 'रुपन्देही' }, { district_name: 'पूर्वी रुकुम' }] },
            { province_id: 6, province_name: 'कर्णाली प्रदेश', districts: [{ district_name: 'दैलेख' }, { district_name: 'डोल्पा' }, { district_name: 'हुम्ला' }, { district_name: 'जाजरकोट' }, { district_name: 'जुम्ला' }, { district_name: 'कालिकोट' }, { district_name: 'मुगु' }, { district_name: 'सल्यान' }, { district_name: 'सुर्खेत' }, { district_name: 'पश्चिम रुकुम' }] },
            { province_id: 7, province_name: 'सुदूरपश्चिम प्रदेश', districts: [{ district_name: 'अछाम' }, { district_name: 'बैतडी' }, { district_name: 'बझाङ' }, { district_name: 'बाजुरा' }, { district_name: 'डडेल्धुरा' }, { district_name: 'दार्चुला' }, { district_name: 'डोटी' }, { district_name: 'कैलाली' }, { district_name: 'कञ्चनपुर' }] }
        ]
    }
};

async function loadInitialData() {
    await Promise.all([loadDocumentTypes(), loadLocationData()]);
}

function ensureLocationData() {
    if (locationData && locationData.country && locationData.country.provinces && locationData.country.provinces.length > 0) {
        return Promise.resolve();
    }
    if (!locationDataPromise) locationDataPromise = loadLocationData();
    return locationDataPromise;
}

async function loadLocationData() {
    try {
        var response = await fetch(API_BASE + '/locations');
        if (!response.ok) { locationData = NEPAL_LOCATIONS_FALLBACK; return; }
        var data = await response.json();
        if (data && data['देश'] && data['देश']['प्रदेशहरू']) {
            var raw = data['देश'];
            locationData = {
                country: {
                    provinces: (raw['प्रदेशहरू'] || []).map(function (p) {
                        return {
                            province_id: p['प्रदेश_आईडी'],
                            province_name: p['प्रदेश_नाम'],
                            districts: (p['जिल्लाहरू'] || []).map(function (d) {
                                var municipalities = (d['स्थानीय_तहहरू'] || []).map(function (m) {
                                    // Handle both plain string entries and object entries
                                    if (typeof m === 'string') {
                                        return { name: m };
                                    }
                                    return { name: m.name || m['नाम'] || '', type: m.type || m['प्रकार'], wards: m.wards || m['वडा'] };
                                });
                                return { district_name: d['जिल्ला_नाम'], municipalities: municipalities };
                            })
                        };
                    })
                }
            };
        } else if (data && data.country && data.country.provinces) {
            locationData = data;
        } else {
            locationData = NEPAL_LOCATIONS_FALLBACK;
        }
        if (!locationData.country || !locationData.country.provinces || locationData.country.provinces.length === 0) {
            locationData = NEPAL_LOCATIONS_FALLBACK;
        }
        console.log('Location data loaded, provinces:', locationData.country.provinces.length);
    } catch (error) {
        console.error('Error loading location data:', error);
        locationData = NEPAL_LOCATIONS_FALLBACK;
    }
}

// =====================================================
//  FORM FIELD GENERATION (ENHANCED WITH TOOLBARS)
// =====================================================
function generateFormFields(fields) {
    // Clear old field states when regenerating
    Object.keys(fieldStates).forEach(function (k) { delete fieldStates[k]; });

    var formFieldsContainer = document.getElementById('formFields');
    if (!formFieldsContainer) return;
    formFieldsContainer.innerHTML = '';
    if (!Array.isArray(fields) || fields.length === 0) return;

    try {
        generateFormFieldsInner(fields, formFieldsContainer);
    } catch (err) {
        console.error('Error generating form fields:', err);
        formFieldsContainer.innerHTML = '<p class="text-red-600">फारम लोड गर्दा त्रुटि। कृपया पृष्ठ रिफ्रेस गर्नुहोस्।</p>';
    }
}

function generateFormFieldsInner(fields, formFieldsContainer) {
    var addressFields = fields.filter(function (f) {
        return f.id.includes('address') || f.id.includes('province') || f.id.includes('district') || f.id.includes('municipality') || f.id.includes('ward');
    });
    var otherFields = fields.filter(function (f) { return !addressFields.includes(f); });

    otherFields.forEach(function (field) { createFieldElement(field, formFieldsContainer); });

    if (addressFields.length > 0) {
        addressFields.forEach(function (field) { createFieldElement(field, formFieldsContainer); });

        var hasPermanent = addressFields.some(function (f) { return f.id.indexOf('permanent') !== -1 || f.id.indexOf('old') !== -1; });
        var hasTemporary = addressFields.some(function (f) { return f.id.indexOf('temporary') !== -1 || f.id.indexOf('new') !== -1; });

        if (hasPermanent && hasTemporary) {
            var copyBtn = document.createElement('button');
            copyBtn.type = 'button';
            copyBtn.className = 'mt-2 mb-4 bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 transition text-sm';
            copyBtn.innerHTML = '<i class="fas fa-copy mr-2"></i>स्थायी ठेगाना नै राख्नुहोस्';
            copyBtn.onclick = copyPermanentToTemporary;
            if (formFieldsContainer.lastElementChild) {
                formFieldsContainer.insertBefore(copyBtn, formFieldsContainer.lastElementChild);
            } else {
                formFieldsContainer.appendChild(copyBtn);
            }
        }
    }
}

function copyPermanentToTemporary() {
    var inputs = document.querySelectorAll('#documentForm input, #documentForm select, #documentForm textarea');
    inputs.forEach(function (input) {
        if (input.id.includes('permanent') || input.id.includes('old')) {
            var targetId = input.id.replace('permanent', 'temporary').replace('old', 'new');
            var targetInput = document.getElementById(targetId);
            if (targetInput) {
                targetInput.value = input.value;
                if (targetInput.tagName === 'SELECT') targetInput.dispatchEvent(new Event('change'));
                // Sync field state
                var state = fieldStates[targetId];
                if (state) state.value = targetInput.value;
            }
        }
    });
}

/**
 * CREATE FIELD ELEMENT — Enhanced with per-field input toolbar.
 * Text/textarea fields get [Keyboard | Mic | Pen] buttons.
 * Each button writes to the same FieldState.
 */
function createFieldElement(field, container) {
    var fieldDiv = document.createElement('div');
    fieldDiv.className = 'mb-5';

    var label = document.createElement('label');
    label.className = 'block text-gray-700 font-semibold mb-1 ' + (field.required ? 'required-field' : '');
    label.textContent = field.label;
    label.setAttribute('for', field.id);

    // Determine if this is a text-type field that should get the input toolbar
    var isTextField = (field.type === 'text' || field.type === 'textarea') && field.type !== 'select';
    
    // Check if this field should skip transliteration (English fields)
    var skipTranslit = isEnglishField(field.id);

    var input;
    if (field.type === 'select') {
        input = document.createElement('select');
        input.className = 'w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500';
        input.id = field.id;
        input.name = field.id;
        input.required = field.required;

        var defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'छनोट गर्नुहोस्';
        input.appendChild(defaultOption);

        if (field.id.indexOf('province') !== -1) {
            if (locationData && locationData.country && locationData.country.provinces && locationData.country.provinces.length > 0) {
                locationData.country.provinces.forEach(function (p) {
                    var option = document.createElement('option');
                    option.value = p.province_name;
                    option.textContent = p.province_name;
                    if (p.province_id != null) option.dataset.id = p.province_id;
                    input.appendChild(option);
                });
            }
            input.addEventListener('change', function (e) { handleProvinceChange(e, field.id); });
        } else if (field.id.indexOf('district') !== -1) {
            if (field.options && field.options.length > 0) {
                field.options.forEach(function (opt) {
                    var o = document.createElement('option'); o.value = opt; o.textContent = opt; input.appendChild(o);
                });
            }
            input.addEventListener('change', function (e) { handleDistrictChange(e, field.id); });
        } else if (field.id.indexOf('municipality') !== -1 || field.id.indexOf('local_body') !== -1) {
            var infoOpt = document.createElement('option');
            infoOpt.value = '';
            infoOpt.textContent = 'पहिले जिल्ला छनोट गर्नुहोस्';
            input.appendChild(infoOpt);
        } else if (field.options) {
            field.options.forEach(function (option) {
                var o = document.createElement('option'); o.value = option; o.textContent = option; input.appendChild(o);
            });
        }
    } else if (field.type === 'textarea') {
        input = document.createElement('textarea');
        input.className = 'w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500';
        input.rows = 4;
        input.id = field.id;
        input.name = field.id;
        input.required = field.required;
    } else {
        input = document.createElement('input');
        input.className = 'w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500';
        input.type = field.type || 'text';
        input.id = field.id;
        input.name = field.id;
        input.required = field.required;
        if (field.type === 'date') input.placeholder = 'YYYY-MM-DD';
    }

    fieldDiv.appendChild(label);

    // ===== PER-FIELD TOOLBAR (for text/textarea fields only) =====
    if (isTextField) {
        var toolbar = createFieldToolbar(field.id);
        fieldDiv.appendChild(toolbar);
        fieldDiv.appendChild(input);

        // Transliteration hint (shows preview as user types English) - skip for English fields
        if (!skipTranslit) {
            var hint = document.createElement('div');
            hint.id = 'hint_' + field.id;
            hint.className = 'transliteration-hint';
            hint.style.display = 'none';
            fieldDiv.appendChild(hint);
        }

        // Register in field state store
        getOrCreateFieldState(field.id, input, field.label);

        // Setup inline keyboard transliteration - skip for English fields
        if (!skipTranslit) {
            setupFieldTransliteration(field.id, input);
        }
    } else {
        fieldDiv.appendChild(input);
    }

    container.appendChild(fieldDiv);
}

/** Create the [Keyboard | Mic | Pen] toolbar for a field */
function createFieldToolbar(fieldId) {
    var toolbar = document.createElement('div');
    toolbar.className = 'field-toolbar';
    toolbar.id = 'toolbar_' + fieldId;

    var modes = [
        { mode: 'typing', icon: 'fa-keyboard', title: 'किबोर्ड' },
        { mode: 'voice', icon: 'fa-microphone', title: 'आवाज' },
        { mode: 'writing', icon: 'fa-pen', title: 'हस्तलेखन' }
    ];

    modes.forEach(function (m) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = m.mode === 'typing' ? 'active' : '';
        btn.dataset.mode = m.mode;
        btn.dataset.field = fieldId;
        btn.innerHTML = '<i class="fas ' + m.icon + '"></i> <span class="text-xs">' + m.title + '</span>';
        btn.title = m.title;
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            handleFieldToolbarClick(fieldId, m.mode);
        });
        toolbar.appendChild(btn);
    });

    return toolbar;
}

function handleFieldToolbarClick(fieldId, mode) {
    var state = fieldStates[fieldId];
    if (!state) return;

    if (mode === 'voice') {
        if (state.isRecording) {
            stopFieldVoiceInput(fieldId);
        } else {
            startFieldVoiceInput(fieldId);
        }
    } else if (mode === 'writing') {
        openFieldCanvas(fieldId);
    } else {
        setFieldMode(fieldId, 'typing');
        if (state.element) {
            state.element.focus();
        }
    }
}

function setFieldMode(fieldId, mode) {
    var state = fieldStates[fieldId];
    if (state) state.activeInputMode = mode;

    var toolbar = document.getElementById('toolbar_' + fieldId);
    if (toolbar) {
        toolbar.querySelectorAll('button').forEach(function (btn) {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    }
}

// =====================================================
//  PER-FIELD KEYBOARD TRANSLITERATION
//  Type English → auto-convert to Nepali on Space/Enter
// =====================================================
function setupFieldTransliteration(fieldId, input) {
    var composingWord = '';
    var composingWordStart = -1;

    input.addEventListener('keydown', function (e) {
        if (e.key === ' ' || e.key === 'Enter') {
            if (composingWord) {
                e.preventDefault();
                var nepali = clientTransliterate(composingWord);
                var val = input.value;
                var cursorPos = input.selectionStart;
                var beforeCursor = val.substring(0, cursorPos);
                var afterCursor = val.substring(cursorPos);

                var wordStart = beforeCursor.lastIndexOf(composingWord);
                if (wordStart !== -1) {
                    var suffix = e.key === ' ' ? ' ' : '\n';
                    input.value = beforeCursor.substring(0, wordStart) + nepali + suffix + afterCursor;
                    var newPos = wordStart + nepali.length + suffix.length;
                    input.setSelectionRange(newPos, newPos);
                }

                composingWord = '';
                composingWordStart = -1;
                hideTranslitHint(fieldId);

                var state = fieldStates[fieldId];
                if (state) state.value = input.value;
            }
        }
    });

    input.addEventListener('input', function () {
        var val = input.value;
        var cursorPos = input.selectionStart;

        // Sync to field state
        var state = fieldStates[fieldId];
        if (state) state.value = val;

        // Find current word (from last space/newline to cursor)
        var beforeCursor = val.substring(0, cursorPos);
        var lastBreak = Math.max(beforeCursor.lastIndexOf(' '), beforeCursor.lastIndexOf('\n'));
        var currentWord = beforeCursor.substring(lastBreak + 1);

        if (currentWord && /^[a-zA-Z]+$/.test(currentWord)) {
            composingWord = currentWord;
            composingWordStart = lastBreak + 1; // Track the start position
            var preview = clientTransliterate(currentWord);
            showTranslitHint(fieldId, preview);
        } else {
            composingWord = '';
            composingWordStart = -1;
            hideTranslitHint(fieldId);
        }
    });

    // Convert remaining word on blur using tracked position
    input.addEventListener('blur', function () {
        if (composingWord && composingWordStart !== -1) {
            var nepali = clientTransliterate(composingWord);
            var val = input.value;
            // Verify the word is still at the tracked position
            if (val.substring(composingWordStart, composingWordStart + composingWord.length) === composingWord) {
                input.value = val.substring(0, composingWordStart) + nepali + val.substring(composingWordStart + composingWord.length);
            }
            composingWord = '';
            composingWordStart = -1;
            hideTranslitHint(fieldId);
            var state = fieldStates[fieldId];
            if (state) state.value = input.value;
        }
    });
}

function showTranslitHint(fieldId, text) {
    var hint = document.getElementById('hint_' + fieldId);
    if (hint) { hint.textContent = '→ ' + text; hint.style.display = 'block'; }
}

function hideTranslitHint(fieldId) {
    var hint = document.getElementById('hint_' + fieldId);
    if (hint) hint.style.display = 'none';
}

// =====================================================
//  PER-FIELD VOICE INPUT
//  Click mic on any field → record → transcribe → fill
// =====================================================
async function startFieldVoiceInput(fieldId) {
    var state = fieldStates[fieldId];
    if (!state || state.isRecording) return;

    try {
        var stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        var mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4'];
        var selectedMime = '';
        for (var i = 0; i < mimeTypes.length; i++) {
            if (MediaRecorder.isTypeSupported(mimeTypes[i])) { selectedMime = mimeTypes[i]; break; }
        }

        var options = selectedMime ? { mimeType: selectedMime } : {};
        state.mediaRecorder = new MediaRecorder(stream, options);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = function (e) { state.audioChunks.push(e.data); };
        state.mediaRecorder.onstop = function () { handleFieldRecordingStop(fieldId); };

        state.mediaRecorder.start();
        state.isRecording = true;

        setFieldMode(fieldId, 'voice');

        // Visual feedback — pulse the input border red
        if (state.element) state.element.classList.add('field-recording');

        // Update toolbar mic button
        var toolbar = document.getElementById('toolbar_' + fieldId);
        if (toolbar) {
            var voiceBtn = toolbar.querySelector('[data-mode="voice"]');
            if (voiceBtn) voiceBtn.innerHTML = '<i class="fas fa-stop text-red-500"></i> <span class="text-xs">बन्द</span>';
        }

        showToast('रेकर्डिङ सुरु भयो — बोल्नुहोस्', 'info', 2000);
    } catch (error) {
        console.error('Mic error:', error);
        showError('माइक्रोफोन पहुँच गर्न सकेन। कृपया ब्राउजर अनुमति जाँच गर्नुहोस्।');
    }
}

function stopFieldVoiceInput(fieldId) {
    var state = fieldStates[fieldId];
    if (!state || !state.isRecording) return;

    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
        state.mediaRecorder.stream.getTracks().forEach(function (t) { t.stop(); });
    }
    state.isRecording = false;

    if (state.element) state.element.classList.remove('field-recording');

    var toolbar = document.getElementById('toolbar_' + fieldId);
    if (toolbar) {
        var voiceBtn = toolbar.querySelector('[data-mode="voice"]');
        if (voiceBtn) voiceBtn.innerHTML = '<i class="fas fa-microphone"></i> <span class="text-xs">आवाज</span>';
    }

    showToast('प्रक्रिया गरिँदैछ...', 'info', 3000);
}

async function handleFieldRecordingStop(fieldId) {
    var state = fieldStates[fieldId];
    if (!state) return;

    var actualMime = state.mediaRecorder.mimeType || 'audio/webm';
    var ext = actualMime.includes('ogg') ? 'ogg' : actualMime.includes('mp4') ? 'mp4' : 'webm';
    var audioBlob = new Blob(state.audioChunks, { type: actualMime });

    // Verify the blob actually has data
    if (audioBlob.size < 100) {
        showError('रेकर्डिङ खाली छ। कृपया पुनः बोल्नुहोस्।');
        return;
    }

    var audioFile = new File([audioBlob], 'recording_' + fieldId + '.' + ext, { type: actualMime });

    showLoading();

    try {
        var fd = new FormData();
        fd.append('audio', audioFile);

        console.log('[Voice] Sending audio for transcription:', audioFile.name, 'size:', audioFile.size, 'type:', actualMime);

        var response = await fetch(API_BASE + '/transcribe-audio', { method: 'POST', body: fd });
        var result = await response.json();

        console.log('[Voice] Server response:', result);

        if (response.ok && result.transcription) {
            // Write directly to the field's state → updates DOM
            var text = result.transcription.trim();
            if (text) {
                state.setValue(text, 'voice');
                showSuccess('आवाज पहिचान सफल!' + (result.grammar_corrected ? ' (व्याकरण सच्याइयो)' : ''));
            } else {
                showError('आवाज पहिचान खाली छ। कृपया स्पष्ट रूपमा बोल्नुहोस्।');
            }
        } else {
            var detail = result.detail || 'Transcription failed';
            console.error('[Voice] Transcription failed:', detail);
            showError('आवाज पहिचान गर्न सकेन: ' + detail);
        }
    } catch (error) {
        console.error('[Voice] Transcription error for field', fieldId, ':', error);
        showError('आवाज पहिचान गर्न सकेन। सर्भर जडान जाँच गर्नुहोस्।');
    } finally {
        hideLoading();
    }
}

// =====================================================
//  PER-FIELD HANDWRITING CANVAS (Modal)
//  Click pen on a field → modal canvas → recognize → fill
// =====================================================
function openFieldCanvas(fieldId) {
    activeCanvasFieldId = fieldId;
    var state = fieldStates[fieldId];

    var labelEl = document.getElementById('canvasFieldLabel');
    if (labelEl) labelEl.textContent = state ? state.label : fieldId;

    clearModalCanvas();

    var modal = document.getElementById('canvasModal');
    if (modal) modal.classList.remove('hidden');

    setFieldMode(fieldId, 'writing');
}

function closeFieldCanvas() {
    var modal = document.getElementById('canvasModal');
    if (modal) modal.classList.add('hidden');
    // Reset the field back to typing mode so user can continue
    if (activeCanvasFieldId) {
        setFieldMode(activeCanvasFieldId, 'typing');
    }
    activeCanvasFieldId = null;
    // Reset alternatives UI
    resetAlternativesUI();
}

function clearModalCanvas() {
    var canvas = document.getElementById('modalCanvas');
    if (canvas) {
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    // Also clear stroke capture for offline recognition
    if (typeof OfflineHandwriting !== 'undefined') {
        OfflineHandwriting.clearCanvasStrokes('modalCanvas');
    }
    // Reset alternatives when canvas is cleared
    resetAlternativesUI();
}

async function submitFieldCanvas() {
    if (!activeCanvasFieldId) return;
    var canvas = document.getElementById('modalCanvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var hasContent = false;
    
    // Check for any pixel with alpha > 0 (meaning something was drawn)
    // Previous bug: was checking for non-black pixels, but user draws in black!
    for (var i = 0; i < imgData.data.length; i += 4) {
        // imgData.data[i+3] is the alpha channel
        // Any pixel with alpha > 0 means something was drawn there
        if (imgData.data[i + 3] > 10) {
            hasContent = true;
            break;
        }
    }

    if (!hasContent) { showError('कृपया पहिले केही लेख्नुहोस्।'); return; }

    showLoading();
    var fieldId = activeCanvasFieldId;

    try {
        var recognizedText = '';
        var serverResult = null;
        
        // ── Strategy: Try server API first (CNN or Tesseract) ──────────────
        try {
            var imageData64 = canvas.toDataURL('image/png');
            console.log('[Handwriting] Sending to server API...');

            var response = await fetch(API_BASE + '/recognize-handwriting', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData64 })
            });

            serverResult = await response.json();
            console.log('[Handwriting] Server response:', serverResult);

            if (response.ok && serverResult.text && serverResult.text.trim().length > 0) {
                recognizedText = serverResult.text.trim();
                lastRecognitionResult = serverResult;
                
                // If CNN returned alternatives, show them for user selection
                if (serverResult.method === 'cnn' && serverResult.alternatives && serverResult.alternatives.length > 0) {
                    hideLoading();
                    showAlternatives(serverResult);
                    return; // Wait for user to select
                }
            }
        } catch (serverError) {
            console.warn('[Handwriting] Server API unavailable:', serverError.message);
        }
        
        // ── Fallback: Client-side Tesseract.js ──────────────────────────────
        if (!recognizedText && typeof TesseractHandwriting !== 'undefined' && TesseractHandwriting.isAvailable()) {
            try {
                console.log('[Handwriting] Using Tesseract.js OCR (fallback)');
                showToast('हस्तलेख पहिचान गर्दै... (पहिलो पटक केही समय लाग्छ)', 'info');
                
                var ocrResult = await TesseractHandwriting.recognize(canvas);
                if (ocrResult && ocrResult.success && ocrResult.text && ocrResult.text.length > 0) {
                    recognizedText = ocrResult.text.trim();
                    console.log('[Handwriting] Tesseract.js result:', recognizedText, 'confidence:', ocrResult.confidence);
                } else if (ocrResult && !ocrResult.success) {
                    console.warn('[Handwriting] Tesseract.js failed:', ocrResult.error);
                }
            } catch (ocrError) {
                console.warn('[Handwriting] Tesseract.js error:', ocrError);
            }
        }
        
        // ── Apply result to field ───────────────────────────────────────────
        if (recognizedText) {
            await applyRecognizedText(recognizedText, fieldId);
        } else {
            showError('हस्तलेख पहिचान गर्न सकेन। कृपया स्पष्ट लेख्नुहोस् वा किबोर्ड प्रयोग गर्नुहोस्।');
        }
    } catch (error) {
        console.error('[Handwriting] Error:', error);
        showError('हस्तलेख पहिचान गर्न सकेन। सर्भर जडान जाँच गर्नुहोस्।');
    } finally {
        hideLoading();
    }
}

/** Show CNN alternatives for user selection */
function showAlternatives(result) {
    var section = document.getElementById('alternativesSection');
    var list = document.getElementById('alternativesList');
    var badge = document.getElementById('recognitionConfidence');
    var recognizeBtn = document.getElementById('recognizeBtn');
    var confirmBtn = document.getElementById('confirmWordBtn');
    
    if (!section || !list) return;
    
    // Build alternatives array: main result + top alternatives
    pendingAlternatives = [{ word: result.text, confidence: result.confidence }];
    if (result.alternatives) {
        result.alternatives.forEach(function(alt) {
            if (alt.word !== result.text) {
                pendingAlternatives.push(alt);
            }
        });
    }
    
    // Show confidence badge
    var conf = Math.round((result.confidence || 0) * 100);
    var confClass = conf >= 70 ? 'confidence-high' : (conf >= 40 ? 'confidence-medium' : 'confidence-low');
    badge.className = 'confidence-badge ' + confClass;
    badge.textContent = conf + '%';
    
    // Render alternative buttons
    list.innerHTML = '';
    pendingAlternatives.forEach(function(alt, idx) {
        var btn = document.createElement('button');
        btn.className = 'alternative-btn' + (idx === 0 ? ' selected' : '');
        btn.textContent = alt.word;
        btn.title = Math.round((alt.confidence || 0) * 100) + '% विश्वास';
        btn.onclick = function() { selectAlternative(idx); };
        list.appendChild(btn);
    });
    
    selectedWordIndex = 0;
    
    // Show/hide buttons
    section.classList.remove('hidden');
    if (recognizeBtn) recognizeBtn.classList.add('hidden');
    if (confirmBtn) confirmBtn.classList.remove('hidden');
}

/** User selects an alternative word */
function selectAlternative(idx) {
    selectedWordIndex = idx;
    var list = document.getElementById('alternativesList');
    if (!list) return;
    
    Array.from(list.children).forEach(function(btn, i) {
        btn.classList.toggle('selected', i === idx);
    });
}

/** Confirm the selected word and apply to field */
async function confirmSelectedWord() {
    if (pendingAlternatives.length === 0) return;
    
    var selectedWord = pendingAlternatives[selectedWordIndex].word;
    var fieldId = activeCanvasFieldId;
    
    showLoading();
    await applyRecognizedText(selectedWord, fieldId);
    hideLoading();
    
    // Reset UI
    resetAlternativesUI();
    closeFieldCanvas();
}

/** Apply recognized text to field with grammar correction */
async function applyRecognizedText(text, fieldId) {
    // Skip grammar correction for English fields
    var correctedText = text;
    if (!isEnglishField(fieldId)) {
        correctedText = await correctNepaliGrammar(text, fieldId);
    }

    var state = fieldStates[fieldId];
    if (state) {
        state.setValue(correctedText, 'handwriting');
        console.log('[Handwriting] Field', fieldId, 'set to:', correctedText);
    } else {
        // Fallback: directly set the DOM element
        var inputEl = document.getElementById(fieldId);
        if (inputEl) {
            inputEl.value = correctedText;
            inputEl.dispatchEvent(new Event('change', { bubbles: true }));
            inputEl.dispatchEvent(new Event('input', { bubbles: true }));
            console.log('[Handwriting] Direct DOM set for', fieldId, ':', correctedText);
        }
    }
    showSuccess('हस्तलेख पहिचान सफल!');
    closeFieldCanvas();
}

/** Reset alternatives UI state */
function resetAlternativesUI() {
    var section = document.getElementById('alternativesSection');
    var list = document.getElementById('alternativesList');
    var recognizeBtn = document.getElementById('recognizeBtn');
    var confirmBtn = document.getElementById('confirmWordBtn');
    
    if (section) section.classList.add('hidden');
    if (list) list.innerHTML = '';
    if (recognizeBtn) recognizeBtn.classList.remove('hidden');
    if (confirmBtn) confirmBtn.classList.add('hidden');
    
    pendingAlternatives = [];
    selectedWordIndex = 0;
    lastRecognitionResult = null;
}

/** Call server to correct Nepali grammar */
async function correctNepaliGrammar(text, fieldId) {
    if (!text || !text.trim()) return text;
    try {
        var state = fieldStates[fieldId];
        var context = state ? state.label : '';
        var response = await fetch(API_BASE + '/correct-grammar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, context: context })
        });
        if (response.ok) {
            var result = await response.json();
            if (result.corrected && result.corrected.trim()) {
                return result.corrected.trim();
            }
        }
    } catch (e) {
        console.warn('[Grammar] Correction failed, using original:', e);
    }
    return text;
}

/** Setup drawing on the modal canvas */
function setupModalCanvas() {
    var canvas = document.getElementById('modalCanvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');

    // Setup stroke capture for offline recognition
    if (typeof OfflineHandwriting !== 'undefined') {
        OfflineHandwriting.setupCanvasCapture('modalCanvas');
        console.log('[Handwriting] Stroke capture initialized for offline recognition');
    }

    canvas.addEventListener('mousedown', function (e) {
        isModalDrawing = true;
        var pos = getCanvasPos(canvas, e);
        modalLastX = pos[0]; modalLastY = pos[1];
    });
    canvas.addEventListener('mousemove', function (e) {
        if (!isModalDrawing) return;
        var pos = getCanvasPos(canvas, e);
        ctx.beginPath();
        ctx.moveTo(modalLastX, modalLastY);
        ctx.lineTo(pos[0], pos[1]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
        modalLastX = pos[0]; modalLastY = pos[1];
    });
    canvas.addEventListener('mouseup', function () { isModalDrawing = false; });
    canvas.addEventListener('mouseout', function () { isModalDrawing = false; });

    // Touch events
    canvas.addEventListener('touchstart', function (e) {
        e.preventDefault();
        isModalDrawing = true;
        var pos = getCanvasPos(canvas, e.touches[0]);
        modalLastX = pos[0]; modalLastY = pos[1];
    });
    canvas.addEventListener('touchmove', function (e) {
        e.preventDefault();
        if (!isModalDrawing) return;
        var pos = getCanvasPos(canvas, e.touches[0]);
        ctx.beginPath();
        ctx.moveTo(modalLastX, modalLastY);
        ctx.lineTo(pos[0], pos[1]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
        modalLastX = pos[0]; modalLastY = pos[1];
    });
    canvas.addEventListener('touchend', function () { isModalDrawing = false; });
}

function getCanvasPos(canvas, e) {
    var rect = canvas.getBoundingClientRect();
    var clientX = e.clientX !== undefined ? e.clientX : e.pageX;
    var clientY = e.clientY !== undefined ? e.clientY : e.pageY;
    return [
        (clientX - rect.left) * (canvas.width / rect.width),
        (clientY - rect.top) * (canvas.height / rect.height)
    ];
}

// =====================================================
//  PROVINCE / DISTRICT / MUNICIPALITY CASCADING
// =====================================================
function handleProvinceChange(e, provinceFieldId) {
    var selectedProvinceName = e.target.value;
    var districtFieldId = provinceFieldId.replace('province', 'district');
    var districtSelect = document.getElementById(districtFieldId);
    if (!districtSelect) return;

    districtSelect.innerHTML = '<option value="">छनोट गर्नुहोस्</option>';
    districtSelect.value = '';

    if (!selectedProvinceName || !locationData || !locationData.country || !locationData.country.provinces) return;

    var province = null;
    for (var i = 0; i < locationData.country.provinces.length; i++) {
        if (locationData.country.provinces[i].province_name === selectedProvinceName) { province = locationData.country.provinces[i]; break; }
    }

    if (province && province.districts && province.districts.length > 0) {
        province.districts.forEach(function (d) {
            var option = document.createElement('option');
            option.value = d.district_name; option.textContent = d.district_name;
            districtSelect.appendChild(option);
        });
    }

    var municipalityFieldId = provinceFieldId.replace('province', 'municipality');
    var municipalitySelect = document.getElementById(municipalityFieldId);
    if (municipalitySelect) municipalitySelect.innerHTML = '<option value="">पहिले जिल्ला छनोट गर्नुहोस्</option>';
}

function handleDistrictChange(e, districtFieldId) {
    var selectedDistrictName = e.target.value;
    var municipalityFieldId = districtFieldId.replace('district', 'municipality');
    var municipalitySelect = document.getElementById(municipalityFieldId);
    var provinceFieldId = districtFieldId.replace('district', 'province');
    var provinceSelect = document.getElementById(provinceFieldId);
    var selectedProvinceName = provinceSelect ? provinceSelect.value : '';

    if (!municipalitySelect) return;
    municipalitySelect.innerHTML = '<option value="">छनोट गर्नुहोस्</option>';
    municipalitySelect.value = '';

    if (!selectedDistrictName || !selectedProvinceName || !locationData || !locationData.country || !locationData.country.provinces) return;

    var province = null;
    for (var i = 0; i < locationData.country.provinces.length; i++) {
        if (locationData.country.provinces[i].province_name === selectedProvinceName) { province = locationData.country.provinces[i]; break; }
    }
    if (!province || !province.districts) return;

    var district = null;
    for (var j = 0; j < province.districts.length; j++) {
        if (province.districts[j].district_name === selectedDistrictName) { district = province.districts[j]; break; }
    }

    if (district && district.municipalities && district.municipalities.length > 0) {
        district.municipalities.forEach(function (m) {
            var option = document.createElement('option');
            option.value = m.name;
            option.textContent = m.name + (m.type ? ' (' + m.type + ')' : '');
            if (m.wards) option.dataset.wards = m.wards;
            municipalitySelect.appendChild(option);
        });
    }
}

// =====================================================
//  INPUT METHOD SELECTION — NO LONGER NEEDED
//  All 3 modes are on every field toolbar now.
//  Kept as no-op for backward compat.
// =====================================================
function selectInputMethod(method) {
    selectedInputMethod = method || 'text';
    goToStep(2);
}

// =====================================================
//  FORM SUBMISSION (ENHANCED WITH NEPALI VALIDATION + GRAMMAR)
// =====================================================
async function handleFormSubmit(e) {
    e.preventDefault();

    // Sync all field states from DOM
    Object.keys(fieldStates).forEach(function (fid) { fieldStates[fid].getValue(); });

    var form = e.target;
    var fd = new FormData(form);
    var data = {};
    for (var pair of fd.entries()) { data[pair[0]] = pair[1]; }

    // Validate required fields
    var requiredFields = form.querySelectorAll('[required]');
    var isValid = true;
    requiredFields.forEach(function (field) {
        if (!field.value.trim()) { field.classList.add('border-red-500'); isValid = false; }
        else { field.classList.remove('border-red-500'); }
    });
    if (!isValid) { showError('कृपया सबै आवश्यक फिल्डहरू भर्नुहोस्।'); return; }

    // ===== NEPALI ENFORCEMENT =====
    var nonNepali = validateAllFieldsNepali();
    if (nonNepali.length > 0) {
        showLoading();
        var autoFixed = 0;
        for (var ni = 0; ni < nonNepali.length; ni++) {
            var f = nonNepali[ni];
            var state = fieldStates[f.fieldId];
            if (state) {
                // First try client-side transliteration
                var converted = clientTransliterate(f.value);
                if (isNepaliText(converted)) {
                    state.setValue(converted, 'auto-transliteration');
                    data[f.fieldId] = converted;
                    autoFixed++;
                } else {
                    // Fall back to server-side transliteration (Gemini)
                    try {
                        var resp = await fetch(API_BASE + '/transliterate', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text: f.value, from_lang: 'en', to_lang: 'ne' })
                        });
                        if (resp.ok) {
                            var trResult = await resp.json();
                            if (trResult.transliterated_text && isNepaliText(trResult.transliterated_text)) {
                                state.setValue(trResult.transliterated_text, 'server-transliteration');
                                data[f.fieldId] = trResult.transliterated_text;
                                autoFixed++;
                            }
                        }
                    } catch (trErr) {
                        console.warn('[Transliterate] Server fallback failed for', f.fieldId, trErr);
                    }
                }
            }
        }
        hideLoading();
        if (autoFixed > 0) {
            showToast(autoFixed + ' फिल्ड स्वचालित नेपालीमा रूपान्तरण गरियो।', 'info');
        }
        // Re-check
        var stillBad = validateAllFieldsNepali();
        if (stillBad.length > 0) {
            showError('यी फिल्डहरू नेपालीमा हुनुपर्छ: ' + stillBad.map(function (f) { return f.label; }).join(', '));
            stillBad.forEach(function (f) {
                var el = document.getElementById(f.fieldId);
                if (el) el.classList.add('border-red-500');
            });
            return;
        }
    }

    // Re-sync data after corrections
    fd = new FormData(form);
    data = {};
    for (var pair of fd.entries()) { data[pair[0]] = pair[1]; }

    window.formData = data;
    showDocumentPreview(data);
    goToStep(3);
}

// =====================================================
//  DOCUMENT GENERATION & DOWNLOAD
// =====================================================
async function generatePDF() {
    showLoading();
    try {
        var response = await fetch(API_BASE + '/generate-document', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ document_type: selectedDocument, user_data: window.formData, language: 'ne' })
        });
        var result = await response.json();
        if (response.ok) {
            document.getElementById('downloadBtn').classList.remove('hidden');
            document.getElementById('downloadBtn').onclick = function () { downloadPDF(result.pdf_path); };
            showDocumentPreview(result.content);
            showSuccess('PDF सफलतापूर्वक उत्पन्न भयो!');
        } else {
            throw new Error(result.detail || 'PDF generation failed');
        }
    } catch (error) {
        console.error('PDF error:', error);
        showError('PDF उत्पन्न गर्न सकेन।');
    } finally {
        hideLoading();
    }
}

function downloadPDF(pdfPath) {
    var filename = pdfPath.replace(/^.*[/\\]/, '');
    var link = document.createElement('a');
    link.href = API_BASE + '/download-document/' + filename;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function escapeHtml(text) {
    if (text == null) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function showDocumentPreview(content) {
    var previewDiv = document.getElementById('documentPreview');
    if (typeof content === 'object' && content !== null) {
        var html = '<div class="bg-gray-50 p-6 rounded-lg"><h3 class="text-lg font-semibold mb-4">दस्तावेज पूर्वावलोकन</h3><div class="bg-white p-4 rounded border">';
        html += '<table class="w-full text-sm">';
        for (var key in content) {
            if (content[key]) {
                html += '<tr class="border-b"><td class="py-2 pr-4 font-medium text-gray-600">' + escapeHtml(key) + '</td><td class="py-2">' + escapeHtml(content[key]) + '</td></tr>';
            }
        }
        html += '</table></div></div>';
        previewDiv.innerHTML = html;
    } else {
        var sanitized = String(content || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        previewDiv.innerHTML = '<div class="bg-gray-50 p-6 rounded-lg"><h3 class="text-lg font-semibold mb-4">दस्तावेज पूर्वावलोकन</h3><div id="printablePreview" class="bg-white p-4 rounded border" style="line-height:1.8"><pre class="whitespace-pre-wrap text-sm" style="font-family: \'Noto Sans Devanagari\', \'Mangal\', \'Preeti\', sans-serif;">' + sanitized + '</pre></div></div>';
    }
}

/** Print the preview directly using browser's Print to PDF - preserves Nepali formatting */
function printPreviewAsPDF() {
    var previewContent = document.getElementById('printablePreview');
    if (!previewContent) {
        previewContent = document.getElementById('documentPreview');
    }
    if (!previewContent) {
        showError('कृपया पहिले PDF उत्पन्न गर्नुहोस्।');
        return;
    }
    
    // Create print window with proper styling
    var printWindow = window.open('', '_blank');
    printWindow.document.write('<!DOCTYPE html><html><head>');
    printWindow.document.write('<meta charset="UTF-8">');
    printWindow.document.write('<title>दस्तावेज प्रिन्ट</title>');
    printWindow.document.write('<style>');
    printWindow.document.write('@import url("https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap");');
    printWindow.document.write('* { font-family: "Noto Sans Devanagari", "Mangal", "Arial Unicode MS", sans-serif; }');
    printWindow.document.write('body { padding: 40px; font-size: 12pt; line-height: 1.8; }');
    printWindow.document.write('pre { white-space: pre-wrap; word-wrap: break-word; font-family: inherit; }');
    printWindow.document.write('@media print { body { padding: 20px; } }');
    printWindow.document.write('</style></head><body>');
    printWindow.document.write(previewContent.innerHTML);
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    
    // Wait for fonts to load then print
    setTimeout(function() {
        printWindow.print();
    }, 500);
}

// =====================================================
//  UTILITY FUNCTIONS
// =====================================================
function showLoading() { var el = document.getElementById('loadingOverlay'); if (el) el.classList.remove('hidden'); }
function hideLoading() { var el = document.getElementById('loadingOverlay'); if (el) el.classList.add('hidden'); }

function rateService(rating) {
    var stars = document.querySelectorAll('.fa-star');
    stars.forEach(function (star, index) {
        if (index < rating) { star.classList.remove('text-gray-300'); star.classList.add('text-yellow-400'); }
        else { star.classList.remove('text-yellow-400'); star.classList.add('text-gray-300'); }
    });
}

function submitFeedback() {
    var feedbackText = document.getElementById('feedbackText').value;
    var rating = document.querySelectorAll('.fa-star.text-yellow-400').length;
    console.log('Feedback:', { rating: rating, feedback: feedbackText });
    showSuccess('तपाईंको प्रतिक्रियाको लागि धन्यवाद!');
    document.getElementById('feedbackText').value = '';
    rateService(0);
}

function startNew() {
    selectedDocument = null;
    selectedInputMethod = 'text';
    formData = {};
    // Clear window.formData to prevent stale data in navigation guard
    if (typeof window !== 'undefined') {
        window.formData = null;
    }
    // Clear field states
    Object.keys(fieldStates).forEach(function (k) { delete fieldStates[k]; });

    var form = document.getElementById('documentForm');
    if (form) form.reset();
    var downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) downloadBtn.classList.add('hidden');
    goToStep(1);
}

// =====================================================
//  LEGACY GLOBAL VOICE / CANVAS COMPAT
//  (kept for backward compat with old HTML onclick)
// =====================================================
function toggleRecording() {
    // Find first text field and toggle its voice
    var firstFieldId = Object.keys(fieldStates)[0];
    if (firstFieldId) {
        var state = fieldStates[firstFieldId];
        if (state.isRecording) stopFieldVoiceInput(firstFieldId);
        else startFieldVoiceInput(firstFieldId);
    }
}
function clearCanvas() { clearModalCanvas(); }
function recognizeHandwriting() { submitFieldCanvas(); }

// =====================================================
//  EXPOSE ALL HANDLERS FOR HTML onclick
// =====================================================
window.selectDocument = selectDocument;
window.selectInputMethod = selectInputMethod;
window.toggleRecording = toggleRecording;
window.clearCanvas = clearCanvas;
window.recognizeHandwriting = recognizeHandwriting;
window.generatePDF = generatePDF;
window.downloadPDF = downloadPDF;
window.printPreviewAsPDF = printPreviewAsPDF;
window.startNew = startNew;
window.rateService = rateService;
window.submitFeedback = submitFeedback;
window.openFieldCanvas = openFieldCanvas;
window.closeFieldCanvas = closeFieldCanvas;
window.clearModalCanvas = clearModalCanvas;
window.submitFieldCanvas = submitFieldCanvas;
window.selectAlternative = selectAlternative;
window.confirmSelectedWord = confirmSelectedWord;
window.goToStep = goToStep;
window.goToStepSafe = goToStepSafe;
window.goBackToForm = goBackToForm;
