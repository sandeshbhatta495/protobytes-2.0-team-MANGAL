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