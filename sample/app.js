// Load settings from CLIENT_CONFIG (defined in config.js)
const WS_URL = CLIENT_CONFIG.websocket.url;
console.log("🚀 WebSocket URL:", WS_URL);

// =====================
// Mic Audio Tran. Config
// =====================
const SAMPLE_RATE = CLIENT_CONFIG.audio.sample_rate;
const BUFFER_SIZE = CLIENT_CONFIG.audio.buffer_size;

const VAD_THRESHOLD = CLIENT_CONFIG.audio.vad_threshold;
const VAD_WINDOW_SIZE = CLIENT_CONFIG.audio.vad_window_size;
const VAD_SILENCE_TIMEOUT = CLIENT_CONFIG.audio.vad_silence_timeout;

const PRE_ROLL_LIMIT = CLIENT_CONFIG.audio.pre_roll_limit;
const MAX_WS_BUFFER_BYTES = CLIENT_CONFIG.audio.max_ws_buffer_bytes;

// =====================
// Mic Audio Tran. State
// =====================
let audioContext = null;
let microphoneStream = null;
let processor = null;
let socket = null;

let isUserSpeaking = false;
let vad_buffer_count = 0;
let silence_counter = 0;
let rms = 0;

let preRollBuffer = [];
let seq = 0;
let lastVoiceSeq = null;

// =====================
// State
// =====================
let isAIProcessing = false;
let isAITurn = false;

// =====================
// Init Audio
// =====================
async function initAudio(ws) {
    socket = ws;

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE
        });
    }

    microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
        }
    });

    const source = audioContext.createMediaStreamSource(microphoneStream);
    processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = onAudioProcess;
}

// =====================
// Audio Process
// =====================
function onAudioProcess(e) {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);

    // -----------------
    // 1. RMS
    // -----------------
    let sum = 0;
    for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
    }
    rms = Math.sqrt(sum / inputData.length);
    document.getElementById('vol-meter').innerText = `VOL: ${rms.toFixed(4)}`;

    // -----------------
    // 2. VAD
    // -----------------
    let justActivated = false;

    if (rms > VAD_THRESHOLD) {
        setListening(true);
        vad_buffer_count++;
        silence_counter = 0;

        if (vad_buffer_count >= VAD_WINDOW_SIZE && !isUserSpeaking) {
            isUserSpeaking = true;
            justActivated = true;
            lastVoiceSeq = null;
            isFirstToken = true;
            currentAIMsgDiv = null;
            if (isAITurn) {
                handleUserBargeIn();
            }
        }
    } else {
        vad_buffer_count = 0;

        if (isUserSpeaking) {
            silence_counter++;
            if (silence_counter >= VAD_SILENCE_TIMEOUT) {
                isUserSpeaking = false;
                setListening(false);
                preRollBuffer = [];

                const currentMode = document.body.classList.contains('aircon-mode') ? 'A' : 'G';

                socket.send(JSON.stringify({
                    type: "end_of_utterance",
                    last_seq: lastVoiceSeq,
                    mode: currentMode
                }));

                isAIProcessing = true;
                isAITurn = true;

                console.log("🎤 End of speech detected and signal sent:", lastVoiceSeq);
            }
        } else {
            setListening(false);
        }
    }

    // -----------------
    // 3. Float32 → Int16 PCM + seq
    // -----------------
    const pcm = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    const buffer = new ArrayBuffer(4 + pcm.byteLength);
    const view = new DataView(buffer);
    view.setUint32(0, seq++, true); // little-endian seq
    new Int16Array(buffer, 4).set(pcm);

    // -----------------
    // 4. Pre-roll flush
    // -----------------
    if (justActivated) {
        while (preRollBuffer.length > 0) {
            if (socket.bufferedAmount > MAX_WS_BUFFER_BYTES) break;
            socket.send(preRollBuffer.shift());

        }
    }

    // -----------------
    // 5. Transmission control
    // -----------------
    if (isUserSpeaking) {
        if (socket.bufferedAmount < MAX_WS_BUFFER_BYTES) {
            socket.send(buffer);
            lastVoiceSeq = seq - 1;
            console.log("🎤 Voice frame sent:", seq);
        } else {
            console.warn("⚠️ WS backpressure:", socket.bufferedAmount);
        }
    } else {
        if (rms > VAD_THRESHOLD / 2) {
            preRollBuffer.push(buffer);
            if (preRollBuffer.length > PRE_ROLL_LIMIT) {
                preRollBuffer.shift();
            }
        }
    }
}

const themeToggle = document.getElementById('theme-toggle');
themeToggle.onchange = (e) => {
    if (e.target.checked) {
        document.body.classList.add('aircon-mode');
    } else {
        document.body.classList.remove('aircon-mode');
    }
};

let nextStartTime = 0, activeSources = [], currentAIMsgDiv = null;
let isFirstToken = true;

const statusDiv = document.getElementById('status');
const chatDisplay = document.getElementById('chat-display');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const visualHub = document.getElementById('visual-hub');

let isConnectedOnce = false;

startBtn.onclick = async () => {
    startBtn.disabled = true;
    isConnectedOnce = false;
    chatDisplay.innerHTML = "";
    addSystemMessage("🎙️ Initializing audio device...");
    statusDiv.innerText = "INITIALIZING...";

    try {
        await initAudio();
        addSystemMessage("🌐 Waiting for server response...");
        statusDiv.innerText = "CONNECTING...";
        connectWebSocket();
    } catch (err) {
        console.error("Initialization failed:", err);
        statusDiv.innerText = "MIC ERROR";
        addSystemMessage("❌ Failed to access microphone. Please check permissions.");
        startBtn.disabled = false;
    }
};

stopBtn.onclick = () => {
    if(socket) socket.close();
    setListening(false);
    setSpeaking(false);
    statusDiv.innerText = "READY";

    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
    }

    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            audioContext = null;
        });
    }
};

function handleUserBargeIn() {
    setListening(true);

    if (isAITurn) {
        console.log("⚡ [Barge-in] AI turn interrupted and control returned to user");

        stopAllPlayback();
        isAIProcessing = false;
        isAITurn = false;
        currentAIMsgDiv = null;
        isFirstToken = true;
        addSystemMessage("✔ Barge-in succeeded", "interrupt-success");
        resetUIState();

        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: "barge_in"
            }));
            console.log("⚡ Sent barge-in signal to server");
        }
    }
}

function resetUIState() {
    setSpeaking(false);
    setListening(false);
    currentAIMsgDiv = null;
    isFirstToken = true;
}

function addChatMessage(content, role) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}-msg`;
    msgDiv.innerText = content;
    chatDisplay.appendChild(msgDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
    return msgDiv;
}

function addSystemMessage(content, type = 'system') {
    const msgDiv = document.createElement('div');
    msgDiv.className = type === 'system' ? 'system-msg' : `interrupt-msg ${type}`;
    msgDiv.innerText = content;
    chatDisplay.appendChild(msgDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
}

function connectWebSocket() {
    socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
        isConnectedOnce = true;
        statusDiv.innerText = "ONLINE";
        chatDisplay.innerHTML = "";
        addSystemMessage("Connected successfully.");
        stopBtn.disabled = false;
    };
    socket.onclose = (event) => {
        statusDiv.innerText = "OFFLINE";
        startBtn.disabled = false;
        stopBtn.disabled = true;
        isAIProcessing = false;
        isAITurn = false;
        stopAllPlayback();
        resetUIState();

        if (!isConnectedOnce) {
            addSystemMessage("⏳ Server is still starting. Please try again shortly.");
        } else {
            addSystemMessage("🔌 Connection to server has been closed.");
        }
    };
    socket.onmessage = async (e) => {
        if (typeof e.data === "string") {
            const msg = JSON.parse(e.data);
            if (msg.type === "interrupt") {
                console.log("⚠️ Received interrupt signal from server");
                stopAllPlayback();
                resetUIState();
                addSystemMessage("🚫 Response interrupted (barge-in)", "system");
                return;
            }
            if (msg.type === "text" && msg.content) {
                if (!isAITurn) {
                    console.warn("🚫 [Text Discard] Dropping leftover text after barge-in");
                    return;
                }
                if (isFirstToken || !currentAIMsgDiv) { currentAIMsgDiv = addChatMessage("", "ai"); isFirstToken = false; }
                currentAIMsgDiv.innerText += msg.content;
                chatDisplay.scrollTop = chatDisplay.scrollHeight;
            }
            if (msg.type === "end_of_response") {
                console.log("✅ AI response completed");
                isAIProcessing = false;
                resetUIState();
            }
        } else { playAudioChunk(e.data); }
    };
}

function stopAllPlayback() {
    activeSources.forEach(s => {
        try {
            s.stop();
            s.disconnect();
        } catch(e) {}
    });
    activeSources = [];
    nextStartTime = 0;
    setSpeaking(false);
    isFirstToken = true;
}

function playAudioChunk(data) {
    if (!isAITurn) {
        console.warn("🚫 [Audio Discard] Dropping leftover chunks after barge-in");
        return;
    }
    const chunkId = Date.now() % 10000;
    console.log(`[Audio Receive] 📥 Chunk received (ID: ${chunkId}, size: ${data.byteLength} bytes)`);
    if (isUserSpeaking && rms > 0.2) {
        console.warn(`[Audio Discard] 🚫 Playback rejected because user is speaking (isUserSpeaking: true) (ID: ${chunkId})`);
        return;
    }
    setSpeaking(true);
    const int16 = new Int16Array(data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
    const buffer = audioContext.createBuffer(1, float32.length, SAMPLE_RATE);
    buffer.getChannelData(0).set(float32);
    const source = audioContext.createBufferSource();
    source.buffer = buffer; source.connect(audioContext.destination);
    const now = audioContext.currentTime;
    if (nextStartTime < now) {
        console.log(`[Audio Timing] 🕒 Buffer underrun, scheduling 50ms later (ID: ${chunkId})`);
        nextStartTime = now + 0.05
    };
    source.start(nextStartTime);
    console.log(`[Audio Schedule] 🔊 Playback scheduled: ${nextStartTime.toFixed(3)}s (ID: ${chunkId}, duration: ${buffer.duration.toFixed(3)}s)`);
    nextStartTime += buffer.duration;
    activeSources.push(source);
    source.onended = () => {
        activeSources = activeSources.filter(s => s !== source);
        if (!isAIProcessing && activeSources.length === 0) {
            console.log(`[Audio Queue] 📭 All playback finished, AI turn completed`);
            isAITurn = false;
            setSpeaking(false);
        }
    };
}

function setListening(a) { visualHub.classList.toggle('is-listening', a); }
function setSpeaking(a) { visualHub.classList.toggle('is-speaking', a); }