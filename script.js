// --- DOM要素の取得 ---
const fileInput = document.getElementById('file-input');
const loadingIndicator = document.getElementById('loading-indicator');
const downloadBtn = document.getElementById('download-btn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video-source');
const statusText = document.getElementById('status'); // ★追加

// --- MediaPipe Vision Taskのインポート (変更なし) ---
const { FaceLandmarker, FilesetResolver } = window;

// --- グローバル変数 ---
let faceLandmarker;
const muzzleImg = new Image();
muzzleImg.src = './cat_muzzle.png';
const FOCAL_LENGTH = 1500;
const Z_SCALE_FACTOR = 800;
const SMOOTHING_FACTOR = 0.6;
let smoothedLandmarksPerFace = [null, null]; 
let animationFrameId;

// --- 初期化処理 (★エラーハンドリングとUIフィードバックを追加) ---
async function initializeFaceLandmarker() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU",
            },
            outputFacialTransformationMatrixes: true,
            runningMode: "VIDEO",
            numFaces: 2,
            minDetectionConfidence: 0.3
        });
        
        // ▼▼▼【変更点】成功時のUI更新 ▼▼▼
        statusText.innerText = "準備ができました。ファイルを選択してください。";
        fileInput.disabled = false;
        console.log("Face Landmarker is ready.");

    } catch (error) {
        // ▼▼▼【変更点】失敗時のUI更新とエラー表示 ▼▼▼
        statusText.innerText = "AIモデルの読み込みに失敗しました。ページを再読み込みしてください。";
        console.error("Failed to initialize Face Landmarker:", error);
        alert(`モデルの読み込みに失敗しました。開発者コンソールで詳細を確認してください。\nError: ${error.message}`);
    }
}
// ページの読み込み完了と同時に初期化を開始
window.addEventListener('load', initializeFaceLandmarker);


// --- イベントリスナー (変更なし) ---
fileInput.addEventListener('change', handleFileSelect);


// --- メイン処理 (変更なし) ---
async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file || !faceLandmarker) return;
    resetUI();
    loadingIndicator.classList.remove('hidden');
    const fileType = file.type.split('/')[0];
    if (fileType === 'image') await processImage(file);
    else if (fileType === 'video') await processVideo(file);
    else { alert('サポートされていないファイル形式です。'); resetUI(); }
}

// (processImage, processVideo, smoothLandmarks, drawWarpedMuzzle, drawTexturedTriangle, resetUI, setupImageDownload の各関数は変更ありませんので、省略します)
// (前回のコードをそのままお使いください)

async function processImage(file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        const results = faceLandmarker.detect(img);
        if (results.faceLandmarks.length > 0) {
            for (let i = 0; i < results.faceLandmarks.length; i++) {
                drawWarpedMuzzle(results.faceLandmarks[i]);
            }
        }
        loadingIndicator.classList.add('hidden');
        setupImageDownload();
    };
}

async function processVideo(file) {
    const videoURL = URL.createObjectURL(file);
    video.src = videoURL;
    video.onloadedmetadata = () => {
        const aspectRatio = video.videoHeight / video.videoWidth;
        canvas.width = video.videoWidth > MAX_PROCESSING_WIDTH ? MAX_PROCESSING_WIDTH : video.videoWidth;
        canvas.height = canvas.width * aspectRatio;
    };
    const stream = canvas.captureStream(30);
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        downloadBtn.href = URL.createObjectURL(blob);
        downloadBtn.download = 'synthesized_video.webm';
        downloadBtn.classList.remove('hidden');
        loadingIndicator.classList.add('hidden');
    };
    recorder.start();
    
    // ▼▼▼【変更点】 video.play()が失敗する可能性を考慮 ▼▼▼
    try {
        await video.play();
        renderLoop();
    } catch (err) {
        alert("動画の再生に失敗しました。ブラウザが自動再生をブロックした可能性があります。");
        console.error("Video play failed:", err);
        resetUI();
        loadingIndicator.classList.add('hidden');
    }

    let lastVideoTime = -1;
    async function renderLoop() {
        if (video.paused || video.ended) {
            if (recorder.state === "recording") recorder.stop();
            cancelAnimationFrame(animationFrameId);
            return;
        }
        if (video.currentTime !== lastVideoTime) {
            const startTimeMs = performance.now();
            const results = faceLandmarker.detectForVideo(video, startTimeMs);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                for (let i = 0; i < results.faceLandmarks.length; i++) {
                    const smoothedLandmarks = smoothLandmarks(results.faceLandmarks[i], i);
                    drawWarpedMuzzle(smoothedLandmarks);
                }
            }
            lastVideoTime = video.currentTime;
        }
        animationFrameId = requestAnimationFrame(renderLoop);
    }
}

function smoothLandmarks(currentLandmarks, faceIndex) {
    let lastKnownLandmarks = smoothedLandmarksPerFace[faceIndex];
    if (!lastKnownLandmarks) {
        smoothedLandmarksPerFace[faceIndex] = currentLandmarks;
        return currentLandmarks;
    }
    const smoothed = [];
    for (let i = 0; i < currentLandmarks.length; i++) {
        const newX = SMOOTHING_FACTOR * currentLandmarks[i].x + (1 - SMOOTHING_FACTOR) * lastKnownLandmarks[i].x;
        const newY = SMOOTHING_FACTOR * currentLandmarks[i].y + (1 - SMOOTHING_FACTOR) * lastKnownLandmarks[i].y;
        const newZ = SMOOTHING_FACTOR * currentLandmarks[i].z + (1 - SMOOTHING_FACTOR) * lastKnownLandmarks[i].z;
        smoothed.push({ x: newX, y: newY, z: newZ, visibility: currentLandmarks[i].visibility });
    }
    smoothedLandmarksPerFace[faceIndex] = smoothed;
    return smoothed;
}

function drawWarpedMuzzle(landmarks) {
    if (!landmarks || muzzleImg.width === 0) return;
    const noseTip = landmarks[4];      
    const noseBridge = landmarks[6];     
    const philtrum = landmarks[13];      
    const leftNostril = landmarks[132]; 
    const rightNostril = landmarks[361];
    const faceWidthRefLeft = landmarks[234];  
    const faceWidthRefRight = landmarks[454]; 
    const center = { x: noseTip.x, y: noseTip.y, z: noseTip.z };
    const vecUp = { x: noseBridge.x - center.x, y: noseBridge.y - center.y, z: noseBridge.z - center.z };
    const vecDown = { x: philtrum.x - center.x, y: philtrum.y - center.y, z: philtrum.z - center.z };
    const vecLeft = { x: leftNostril.x - center.x, y: leftNostril.y - center.y, z: leftNostril.z - center.z };
    const vecRight = { x: rightNostril.x - center.x, y: rightNostril.y - center.y, z: rightNostril.z - center.z };
    const faceWidth = Math.sqrt(Math.pow(faceWidthRefRight.x - faceWidthRefLeft.x, 2) + Math.pow(faceWidthRefRight.y - faceWidthRefLeft.y, 2));
    const scale = faceWidth * 1.5;
    const vertices3D = [
        { x: center.x + vecUp.x + vecLeft.x * scale, y: center.y + vecUp.y + vecLeft.y * scale, z: center.z + vecUp.z + vecLeft.z * scale },
        { x: center.x + vecUp.x,                     y: center.y + vecUp.y,                     z: center.z + vecUp.z },
        { x: center.x + vecUp.x + vecRight.x * scale,y: center.y + vecUp.y + vecRight.y * scale,z: center.z + vecUp.z + vecRight.z * scale },
        { x: center.x + vecLeft.x * scale,           y: center.y + vecLeft.y * scale,           z: center.z + vecLeft.z * scale },
        center,
        { x: center.x + vecRight.x * scale,          y: center.y + vecRight.y * scale,          z: center.z + vecRight.z * scale },
        { x: center.x + vecDown.x + vecLeft.x * scale, y: center.y + vecDown.y + vecLeft.y * scale, z: center.z + vecDown.z + vecLeft.z * scale },
        { x: center.x + vecDown.x,                       y: center.y + vecDown.y,                       z: center.z + vecDown.z },
        { x: center.x + vecDown.x + vecRight.x * scale,y: center.y + vecDown.y + vecRight.y * scale,z: center.z + vecDown.z + vecRight.z * scale }
    ];
    const vertices2D = vertices3D.map(v => {
        const perspective = FOCAL_LENGTH / (FOCAL_LENGTH + v.z * Z_SCALE_FACTOR);
        return {
            x: v.x * canvas.width * perspective,
            y: v.y * canvas.height * perspective
        };
    });
    const triangles = [
        [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]
    ];
    const uv_coords = [
        { u: 0, v: 0 }, { u: 0.5, v: 0 }, { u: 1, v: 0 },
        { u: 0, v: 0.5 }, { u: 0.5, v: 0.5 }, { u: 1, v: 0.5 },
        { u: 0, v: 1 }, { u: 0.5, v: 1 }, { u: 1, v: 1 }
    ];
    for (const tri of triangles) {
        const p0 = vertices2D[tri[0]], p1 = vertices2D[tri[1]], p2 = vertices2D[tri[2]];
        const uv0 = uv_coords[tri[0]], uv1 = uv_coords[tri[1]], uv2 = uv_coords[tri[2]];
        drawTexturedTriangle(p0, p1, p2, uv0, uv1, uv2);
    }
}

function drawTexturedTriangle(p0, p1, p2, uv0, uv1, uv2) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.closePath();
    ctx.clip();
    const t0 = { x: uv0.u * muzzleImg.width, y: uv0.v * muzzleImg.height };
    const t1 = { x: uv1.u * muzzleImg.width, y: uv1.v * muzzleImg.height };
    const t2 = { x: uv2.u * muzzleImg.width, y: uv2.v * muzzleImg.height };
    const delta = t0.x * t1.y + t1.x * t2.y + t2.x * t0.y - t1.x * t0.y - t2.x * t1.y - t0.x * t2.y;
    if (Math.abs(delta) < 1e-6) { ctx.restore(); return; }
    const a = (p0.x * t1.y + p1.x * t2.y + p2.x * t0.y - p1.x * t0.y - p2.x * t1.y - p0.x * t2.y) / delta;
    const b = (p0.y * t1.y + p1.y * t2.y + p2.y * t0.y - p1.y * t0.y - p2.y * t1.y - p0.y * t2.y) / delta;
    const c = (p0.x * t2.x + p1.x * t0.x + p2.x * t1.x - p1.x * t2.x - p2.x * t0.x - p0.x * t1.x) / delta;
    const d = (p0.y * t2.x + p1.y * t0.x + p2.y * t1.x - p1.y * t2.x - p2.y * t0.x - p0.y * t1.x) / delta;
    const e = (p0.x * (t1.y * t2.x - t2.y * t1.x) + p1.x * (t2.y * t0.x - t0.y * t2.x) + p2.x * (t0.y * t1.x - t1.y * t0.x)) / delta;
    const f = (p0.y * (t1.y * t2.x - t2.y * t1.x) + p1.y * (t2.y * t0.x - t0.y * t2.x) + p2.y * (t0.y * t1.x - t1.y * t0.x)) / delta;
    ctx.transform(a, b, c, d, e, f);
    ctx.drawImage(muzzleImg, 0, 0);
    ctx.restore();
}

function resetUI() {
    loadingIndicator.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if(animationFrameId) cancelAnimationFrame(animationFrameId);
    smoothedLandmarksPerFace = [null, null];
}
function setupImageDownload() {
    downloadBtn.href = canvas.toDataURL('image/png');
    downloadBtn.download = 'synthesized_image.png';
    downloadBtn.classList.remove('hidden');
}