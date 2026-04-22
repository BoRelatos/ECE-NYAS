// ===================================================================================
// ECENYAS UNIFIED SCRIPT (SORTED TRANSLATIONS & BILINGUAL SEARCH)
// ===================================================================================

// --- GLOBAL VARIABLES FOR DATA ---
let signsDatabase = {};   
let translationsDB = {};  

// --- DATA LOADER ---
async function loadUnifiedDatabase() {
    try {
        console.log("Fetching unified database...");
        const response = await fetch('/api/get_signs_data');
        if (!response.ok) throw new Error("Failed to load data");
        
        signsDatabase = await response.json();
        
        // Flatten for Sign-to-Text
        for (const category in signsDatabase) {
            signsDatabase[category].forEach(item => {
                const entry = { ...item, ...item.translations };
                if (item.english) translationsDB[item.english.toLowerCase()] = entry;
                if (item.filipino) translationsDB[item.filipino.toLowerCase()] = entry;
            });
        }
        
        console.log("Unified database loaded successfully.");

        // Trigger Explore Page Load if applicable
        if (document.getElementById('signs-list-container')) {
            loadCategory('Alphabet');
        }

    } catch (error) {
        console.error("Error loading unified database:", error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadUnifiedDatabase();
    // Mobile Menu
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));
    }
});

// ===================================================================================
// --- 2. SIGN-TO-TEXT PAGE LOGIC ---
// ===================================================================================
const videoElement = document.getElementById('video-input');
if (videoElement) {
    const canvasElement = document.getElementById('overlay');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const refreshBtn = document.getElementById('refresh-btn');
    const signTextElem = document.getElementById('sign-text');
    const emotionTextElem = document.getElementById('emotion-text');
    const timeTextElem = document.getElementById('prediction-time');
    const langSelectToggle = document.getElementById('lang-select-toggle');
    const cameraPlaceholder = document.getElementById('camera-placeholder');

    // Translation Elements
    const englishTextElem = document.getElementById('english-text');
    const filipinoTextElem = document.getElementById('filipino-text');
    const cebuanoTextElem = document.getElementById('cebuano-text');
    const ilokoTextElem = document.getElementById('iloko-text');
    const warayTextElem = document.getElementById('waray-text');
    const bicolanoTextElem = document.getElementById('bicolano-text');
    const hiligaynonTextElem = document.getElementById('hiligaynon-text');
    const kapampanganTextElem = document.getElementById('kapampangan-text');

    // Status Text & Toast
    const statusText = document.createElement('div');
    statusText.id = 'status-text';
    statusText.style = `position: fixed; top: 12px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); color: #fff; font-family: 'Poppins', sans-serif; padding: 6px 12px; border-radius: 10px; z-index: 9999; font-size: 15px; transition: opacity 0.3s;`;
    statusText.textContent = 'Idle';
    document.body.appendChild(statusText);

    const toast = document.createElement('div');
    toast.id = 'cooldown-toast';
    toast.style = `position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #222; color: #ffd700; padding: 8px 14px; border-radius: 10px; font-family: 'Poppins', sans-serif; font-size: 14px; opacity: 0; transition: opacity 0.4s; z-index: 9999;`;
    document.body.appendChild(toast);

    function showToast(message, duration = 1500) {
        toast.textContent = message;
        toast.style.opacity = 1;
        setTimeout(() => (toast.style.opacity = 0), duration);
    }

    const signApiEndpoint = '/predict';
    const emotionApiEndpoint = '/emotion';
    const SEQUENCE_LENGTH = 45;         
    const MIN_NONEMPTY_RATIO = 0.10;    
    const MAX_EMPTY_FRAMES = 5;         
    const NO_HANDS_BUFFER_MS = 1000;    
    const CLIENT_SMOOTH_WINDOW = 7;     

    let sequence = [];                  
    let clientBuffer = [];              
    let emptyFrames = 0;                
    let startDetectTime = null;         
    let awaitingHandsClear = false;     
    let handsClearedAt = null;          
    let isPredicting = false;           
    let lastPredictionTimestamp = 0;    

    const extractKeypoints = (res) => {
        const getCoords = (hand) => hand ? hand.map(lm => [lm.x, lm.y, lm.z]).flat() : new Array(21 * 3).fill(0);
        return [...getCoords(res.leftHandLandmarks), ...getCoords(res.rightHandLandmarks)];
    };
    const isFrameEmpty = (frame) => frame.every(v => v === 0);
    const compressSequence = (seq) => seq.map(frame => frame.map(v => Number(v.toFixed(4))));

    function updateTranslations(sign) {
        if (!sign) return;
        const lookupKey = sign.toLowerCase().trim();
        const entry = translationsDB[lookupKey];

        if (entry) {
            if (englishTextElem) englishTextElem.innerText = entry.english || '...';
            if (filipinoTextElem) filipinoTextElem.innerText = entry.filipino || '...';
            if (cebuanoTextElem) cebuanoTextElem.innerText = entry.cebuano || '...';
            if (ilokoTextElem) ilokoTextElem.innerText = entry.iloko || entry.ilocano || '...';
            if (warayTextElem) warayTextElem.innerText = entry.waray || '...';
            if (bicolanoTextElem) bicolanoTextElem.innerText = entry.bicolano || '...';
            if (hiligaynonTextElem) hiligaynonTextElem.innerText = entry.hiligaynon || '...';
            if (kapampanganTextElem) kapampanganTextElem.innerText = entry.kapampangan || '...';
        } else {
            [englishTextElem, filipinoTextElem, cebuanoTextElem, ilokoTextElem, warayTextElem, 
             bicolanoTextElem, hiligaynonTextElem, kapampanganTextElem].forEach(el => { if(el) el.innerText = '...'; });
        }
    }
    
    function acceptPrediction(label, conf, predTimeSeconds = null) {
        clientBuffer.push({ label, conf });
        if (clientBuffer.length > CLIENT_SMOOTH_WINDOW) clientBuffer.shift();

        const votes = {};
        clientBuffer.forEach(p => { votes[p.label] = (votes[p.label] || 0) + (p.conf || 0.001); });
        const best = Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);

        if (predTimeSeconds !== null) {
            timeTextElem.innerText = `${predTimeSeconds.toFixed(2)}s`;
        }

        if (best && best !== "No confident prediction" && best !== "...") {
            if (!signTextElem.value.trim().endsWith(best)) {
                signTextElem.value += best + " ";
                updateTranslations(best);
            }
            clientBuffer = [];
            awaitingHandsClear = true;
            handsClearedAt = null;
            statusText.textContent = "Hand sign accepted";
            statusText.style.color = "#ffd700";
            showToast("Waiting for hands to clear…");
        }
    }

    async function sendToServer(seqArray, language) {
        try {
            const resp = await fetch(signApiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: compressSequence(seqArray), language })
            });
            return resp.ok ? await resp.json() : { sign: null, confidence: 0 };
        } catch { return { sign: null, confidence: 0 }; }
    }

    async function handlePrediction(seqCopy) {
        if (isPredicting || awaitingHandsClear) { isPredicting = false; return; }
        isPredicting = true;
        const nonEmptyCount = seqCopy.filter(f => !isFrameEmpty(f)).length;
        if (nonEmptyCount < SEQUENCE_LENGTH * MIN_NONEMPTY_RATIO) { isPredicting = false; return; }

        statusText.textContent = "Analyzing...";
        statusText.style.color = "#ffd700";
        const language = langSelectToggle && langSelectToggle.checked ? 'ASL' : 'FSL';
        
        const startTime = performance.now(); 
        const res = await sendToServer(seqCopy, language);
        const endTime = performance.now();
        
        if (res && res.sign && res.sign !== "No confident prediction") {
            acceptPrediction(res.sign, res.confidence || 0, (endTime - startTime) / 1000);
            lastPredictionTimestamp = Date.now();
        }
        isPredicting = false;
    }

    function onResults(results) {
        const ctx = canvasElement.getContext('2d');
        ctx.save();
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        ctx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        
        const keypoints = extractKeypoints(results);
        if (isFrameEmpty(keypoints)) {
            emptyFrames++;
            if (awaitingHandsClear && emptyFrames > MAX_EMPTY_FRAMES && handsClearedAt === null) handsClearedAt = performance.now();
            statusText.textContent = "No hand sign to detect";
            statusText.style.color = "#ccc";
        } else {
            emptyFrames = 0;
            if (awaitingHandsClear) { awaitingHandsClear = false; handsClearedAt = null; clientBuffer = []; }
            if (startDetectTime === null) startDetectTime = performance.now();
            sequence.push(keypoints);
            if (sequence.length > SEQUENCE_LENGTH) sequence = sequence.slice(-SEQUENCE_LENGTH);
            statusText.textContent = "Analyzing...";
            statusText.style.color = "#ffd700";
        }

        if (awaitingHandsClear && handsClearedAt && (performance.now() - handsClearedAt > NO_HANDS_BUFFER_MS)) {
            awaitingHandsClear = false; handsClearedAt = null; sequence = []; clientBuffer = [];
        }

        if (sequence.length === SEQUENCE_LENGTH && !awaitingHandsClear && (Date.now() - lastPredictionTimestamp > 800)) {
            const seqCopy = [...sequence];
            setTimeout(() => handlePrediction(seqCopy), 0);
        }
        ctx.restore();
    }

    function detectEmotion() {
        try {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 160; tempCanvas.height = videoElement.videoHeight * (160 / videoElement.videoWidth);
            tempCanvas.getContext('2d').drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
            
            fetch(emotionApiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: tempCanvas.toDataURL('image/jpeg', 0.5) })
            }).then(r => r.json()).then(data => {
                if (data.emotion && data.emotion !== 'none') emotionTextElem.innerText = data.emotion;
            }).catch(() => {});
        } catch {}
    }

    const holistic = new window.Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}` });
    holistic.setOptions({ minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    holistic.onResults(onResults);

    startCameraBtn.addEventListener('click', () => {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            showToast('Browser does not support camera access.'); return;
        }
        sequence = []; clientBuffer = []; emptyFrames = 0;
        signTextElem.value = ''; 
        if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
        videoElement.classList.remove('hidden'); 

        const camera = new Camera(videoElement, { 
            onFrame: async () => await holistic.send({ image: videoElement })
        });

        camera.start().then(() => setInterval(detectEmotion, 1500)).catch(() => {
            showToast('Camera access failed.');
            videoElement.classList.add('hidden'); 
            if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex'; 
        });
    });

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            sequence = []; clientBuffer = []; emptyFrames = 0; awaitingHandsClear = false; handsClearedAt = null;
            signTextElem.value = ''; emotionTextElem.innerText = '...'; timeTextElem.innerText = '...';
            updateTranslations(null);
            statusText.textContent = 'Reset complete';
        });
    }
}

// ===================================================================================
// --- 3. SPEECH-TO-SIGN PAGE LOGIC ---
// ===================================================================================
const speechTextarea = document.getElementById('speech-text');
if (speechTextarea) {
    const startMicBtn = document.getElementById('start-mic-btn');
    const stopMicBtn = document.getElementById('stop-mic-btn'); // Can be hidden
    const translateBtn = document.getElementById('translate-btn');
    const refreshMicBtn = document.getElementById('refresh-mic-btn');
    const translationTimeElem = document.getElementById('translation-time');

    const translationSpans = {
        english: document.getElementById('english-translation'),
        filipino: document.getElementById('filipino-translation'),
        bicolano: document.getElementById('bicolano-text'),
        cebuano: document.getElementById('cebuano-text'),
        ilocano: document.getElementById('iloko-text'),
        hiligaynon: document.getElementById('hiligaynon-text'),
        kapampangan: document.getElementById('kapampangan-text'),
        waray: document.getElementById('waray-text')
    };

    const aslGif = document.getElementById('asl-gif');
    const fslGif = document.getElementById('fsl-gif');
    let recognition = null;
    let isMicRecording = false;
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.interimResults = true; 
        // Set to fil-PH. This acts as our "Taglish" hack!
        recognition.lang = 'fil-PH'; 

        recognition.onstart = () => { isMicRecording = true; speechTextarea.placeholder = "Listening..."; startMicBtn.disabled = true; };
        recognition.onresult = (event) => { speechTextarea.value = Array.from(event.results).map(result => result[0].transcript).join(''); };
        recognition.onend = () => { isMicRecording = false; speechTextarea.placeholder = "Your speech will appear here..."; startMicBtn.disabled = false; };
        recognition.onerror = () => { isMicRecording = false; speechTextarea.placeholder = "Error, try again or type manually."; startMicBtn.disabled = false; };
    }

    const translateText = async () => {
        const textToTranslate = speechTextarea.value.trim();
        if (!textToTranslate) { alert("Please enter text."); return; }
        translateBtn.disabled = true; translateBtn.textContent = 'Translating...'; translationTimeElem.textContent = '...';

        try {
            const startTime = performance.now();
            const response = await fetch('/translate_text', { 
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: textToTranslate })
            });
            if (!response.ok) throw new Error();
            const data = await response.json();
            
            for (const lang in translationSpans) if(translationSpans[lang]) translationSpans[lang].textContent = data[lang] || 'N/A';
            if(aslGif) aslGif.src = data.asl_gif || 'https://placehold.co/400x300/E0E0E0/333333?text=Error';
            if(fslGif) fslGif.src = data.fsl_gif || 'https://placehold.co/400x300/E0E0E0/333333?text=Error';
            translationTimeElem.textContent = `${((performance.now() - startTime) / 1000).toFixed(2)}s`;
        } catch { alert("Translation failed."); } 
        finally { translateBtn.disabled = false; translateBtn.textContent = 'Translate'; }
    };

    const resetTranslations = () => {
        speechTextarea.value = ''; translationTimeElem.textContent = '...';
        if(aslGif) aslGif.src = 'https://placehold.co/400x300/E0E0E0/333333?text=ASL+Sign';
        if(fslGif) fslGif.src = 'https://placehold.co/400x300/E0E0E0/333333?text=FSL+Sign';
        for (const lang in translationSpans) if(translationSpans[lang]) translationSpans[lang].textContent = '...';
        if (recognition && isMicRecording) recognition.stop();
    };

    translateBtn.addEventListener('click', translateText);
    startMicBtn.addEventListener('click', () => { if (recognition && !isMicRecording) recognition.start(); });
    if(stopMicBtn) stopMicBtn.addEventListener('click', () => { if (recognition && isMicRecording) recognition.stop(); });
    refreshMicBtn.addEventListener('click', resetTranslations);
}

// ===================================================================================
// --- 4. EXPLORE PAGE LOGIC (SPLIT PANE + BILINGUAL + SORTED TRANSLATIONS) ---
// ===================================================================================

const signsListContainer = document.getElementById('signs-list-container');
const signDetailContainer = document.getElementById('sign-detail-container');

if (signsListContainer && signDetailContainer) {
    const categoryLinks = document.querySelectorAll('.category-link');
    const categoryTitle = document.getElementById('current-category-title');
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');

    // 1. RENDER LIST (Left Column)
    function renderSignList(signs) {
        signsListContainer.innerHTML = '';
        if (signs.length === 0) {
            signsListContainer.innerHTML = '<div class="text-brand-lb/50 text-center py-4">No signs found.</div>';
            return;
        }

        signs.forEach(sign => {
            const card = document.createElement('div');
            card.className = "cursor-pointer rounded-xl border border-brand-ol bg-brand-w p-4 hover:border-brand-bu hover:shadow-md transition-all";
            // Display English and Filipino in the list card
            card.innerHTML = `
                <h3 class="font-semibold text-brand-b">${sign.english}</h3>
                <p class="text-xs text-brand-lb/70 italic">${sign.filipino || ''}</p>
            `;
            
            // CLICK EVENT: Load details into Right Column
            card.addEventListener('click', () => {
                // Highlight active card
                document.querySelectorAll('#signs-list-container > div').forEach(c => c.classList.remove('border-brand-bu', 'ring-2', 'ring-brand-bu/20'));
                card.classList.add('border-brand-bu', 'ring-2', 'ring-brand-bu/20');
                renderSignDetail(sign);
            });

            signsListContainer.appendChild(card);
        });
    }

    // 2. RENDER DETAIL (Right Column)
    function renderSignDetail(sign) {
        const transMap = sign.translations || {};
        
        // Define the specific order: English, Filipino, then Alphabetical
        const orderedLangs = ['English', 'Filipino', 'Bicolano', 'Cebuano', 'Hiligaynon', 'Ilocano', 'Kapampangan', 'Waray'];
        
        let translationsHTML = '';
        
        orderedLangs.forEach(lang => {
            let val = '...';
            if (lang === 'English') {
                val = sign.english; // Get directly from sign object
            } else if (lang === 'Filipino') {
                val = sign.filipino; // Get directly from sign object
            } else {
                // Try lowercase or original case key from translations map
                val = transMap[lang.toLowerCase()] || transMap[lang] || '...';
            }
            
            translationsHTML += `<div><span class="font-medium text-brand-b">${lang}:</span> <span class="text-brand-lb">${val}</span></div>`;
        });

        // Inject HTML
        signDetailContainer.innerHTML = `
            <div class="rounded-2xl border-2 border-brand-bu/30 bg-brand-accent/50 p-6 animate-in fade-in zoom-in-95 duration-300">
                <h2 class="text-3xl font-bold text-brand-b mb-6 border-b border-brand-ol pb-4">${sign.english}</h2>
                
                <div class="grid gap-6 lg:grid-cols-2 mb-8">
                    <div class="space-y-2">
                        <h4 class="font-semibold text-center text-brand-b">ASL</h4>
                        <div class="aspect-[3/2] bg-brand-w rounded-xl border border-brand-ol flex items-center justify-center overflow-hidden">
                            <img src="${sign.asl_gif}" class="w-full h-full object-contain" alt="ASL for ${sign.english}">
                        </div>
                    </div>
                    <div class="space-y-2">
                        <h4 class="font-semibold text-center text-brand-b">FSL</h4>
                        <div class="aspect-[3/2] bg-brand-w rounded-xl border border-brand-ol flex items-center justify-center overflow-hidden">
                            <img src="${sign.fsl_gif}" class="w-full h-full object-contain" alt="FSL for ${sign.english}">
                        </div>
                    </div>
                </div>

                <div>
                    <h4 class="font-semibold text-lg text-brand-b mb-3">Translations</h4>
                    <div class="bg-brand-w rounded-xl border border-brand-ol p-5 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm">
                        ${translationsHTML}
                    </div>
                </div>
            </div>
        `;
    }

    // 3. LOAD CATEGORY
    window.loadCategory = function(categoryName) {
        if (!signsDatabase || !signsDatabase[categoryName]) {
            signsListContainer.innerHTML = '<div class="text-brand-lb/50">Category empty or loading...</div>';
            return;
        }
        
        if(categoryTitle) categoryTitle.textContent = categoryName;

        categoryLinks.forEach(link => {
            if (link.dataset.category === categoryName) link.classList.add('active');
            else link.classList.remove('active');
        });

        renderSignList(signsDatabase[categoryName]);
        
        // Reset Right Column
        signDetailContainer.innerHTML = `
            <div class="rounded-2xl border-2 border-brand-bu/30 bg-brand-accent/50 p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand-bu/40 mb-4"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
                <h3 class="text-xl font-semibold text-brand-b">Select a sign</h3>
                <p class="text-brand-lb/70 mt-2">Click on any sign from the list on the left to view its details and translations.</p>
            </div>
        `;
    }

    // 4. SEARCH FUNCTION
    function performSearch(query) {
        if (!signsDatabase) return;
        const normalizedQuery = query.toLowerCase().trim();
        categoryLinks.forEach(link => link.classList.remove('active')); // Deselect categories
        
        if (!normalizedQuery) { loadCategory('Alphabet'); return; }

        let searchResults = [];
        for (const cat in signsDatabase) {
            signsDatabase[cat].forEach(sign => {
                const en = sign.english ? sign.english.toLowerCase() : '';
                const fil = sign.filipino ? sign.filipino.toLowerCase() : '';
                
                if (en.includes(normalizedQuery) || fil.includes(normalizedQuery)) {
                    searchResults.push(sign);
                }
            });
        }

        if(categoryTitle) categoryTitle.textContent = `Search: "${query}"`;
        renderSignList(searchResults);
        signDetailContainer.innerHTML = '<div class="text-center p-12 text-brand-lb/50">Select a result to view details</div>';
    }

    categoryLinks.forEach(link => link.addEventListener('click', (e) => loadCategory(e.target.dataset.category)));
    searchButton.addEventListener('click', () => performSearch(searchInput.value));
    searchInput.addEventListener('keyup', (e) => { if (e.key === 'Enter') performSearch(searchInput.value); });
}