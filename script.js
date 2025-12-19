// --- DOM要素の取得 (変更なし) ---
const fileInput = document.getElementById('file-input');
const loadingIndicator = document.getElementById('loading-indicator');
const downloadBtn = document.getElementById('download-btn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video-source');

// --- MediaPipe Vision Taskのインポート (変更なし) ---
const { FaceLandmarker, FilesetResolver } = window;

// --- 定数とグローバル変数 ---
let faceLandmarker;
const muzzleImg = new Image();
muzzleImg.src = './cat_muzzle.png';

const MAX_PROCESSING_WIDTH = 1280;
const SMOOTHING_FACTOR = 0.6; // 少し追従性を上げて、顔の向きへの反応を良くする

// ▼▼▼ 【今回の主要な変更点】 ▼▼▼
// 3D変形のための定数
const FOCAL_LENGTH = 1500; // 遠近感の強さを調整する焦点距離。値が小さいほどパースが強くなる。
const Z_SCALE_FACTOR = 800;  // MediaPipeのz座標のスケールを調整する係数。

// 顔ごとのスムージングされたランドマーク情報を保持する配列
let smoothedLandmarksPerFace = [null, null]; 
let animationFrameId;
// ▲▲▲ ここまで ▲▲▲


// --- 初期化処理 (minDetectionConfidenceは維持) ---
async function initializeFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU",
        },
        outputFacialTransformationMatrixes: true, // 3D情報を使うために重要
        runningMode: "VIDEO",
        numFaces: 2,
        minDetectionConfidence: 0.3
    });
    console.log("Face Landmarker is ready.");
}
initializeFaceLandmarker();


// --- メインループと画像/動画処理 (renderLoopの呼び出し先を変更する以外はほぼ変更なし) ---
fileInput.addEventListener('change', handleFileSelect);

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
    video.play();
    let lastVideoTime = -1;
    async function renderLoop() {
        if (video.paused || video.ended) {
            recorder.stop();
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
                    // ▼▼▼【変更点】呼び出す描画関数を新しいものに差し替え▼▼▼
                    drawWarpedMuzzle(smoothedLandmarks);
                    // ▲▲▲ ここまで ▲▲▲
                }
            }
            lastVideoTime = video.currentTime;
        }
        animationFrameId = requestAnimationFrame(renderLoop);
    }
    renderLoop();
}

// --- スムージング処理 (変更なし) ---
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


// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
// --- 【新規】メッシュ変形描画のメイン関数 ---
// ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

function drawWarpedMuzzle(landmarks) {
    if (!landmarks || muzzleImg.width === 0) return;

    // 1. マズルメッシュの基準となる主要なランドマークを定義
    const noseTip = landmarks[4];      // 中心
    const noseBridge = landmarks[6];     // 上
    const philtrum = landmarks[13];      // 下
    const leftNostril = landmarks[132]; // 左
    const rightNostril = landmarks[361];// 右
    const faceWidthRefLeft = landmarks[234];  //顔幅基準（左）
    const faceWidthRefRight = landmarks[454]; //顔幅基準（右）

    // 2. 3D空間上でのマズルメッシュの基底ベクトルを計算
    const center = { x: noseTip.x, y: noseTip.y, z: noseTip.z };
    const vecUp = { x: noseBridge.x - center.x, y: noseBridge.y - center.y, z: noseBridge.z - center.z };
    const vecDown = { x: philtrum.x - center.x, y: philtrum.y - center.y, z: philtrum.z - center.z };
    const vecLeft = { x: leftNostril.x - center.x, y: leftNostril.y - center.y, z: leftNostril.z - center.z };
    const vecRight = { x: rightNostril.x - center.x, y: rightNostril.y - center.y, z: rightNostril.z - center.z };
    const faceWidth = Math.sqrt(Math.pow(faceWidthRefRight.x - faceWidthRefLeft.x, 2) + Math.pow(faceWidthRefRight.y - faceWidthRefLeft.y, 2));

    // 3. マズルの3x3グリッド（9つの頂点）を3D空間上に構築
    const scale = faceWidth * 1.5; // マズルの全体的な大きさを調整
    const vertices3D = [
        // Top row
        { x: center.x + vecUp.x + vecLeft.x * scale, y: center.y + vecUp.y + vecLeft.y * scale, z: center.z + vecUp.z + vecLeft.z * scale },
        { x: center.x + vecUp.x,                     y: center.y + vecUp.y,                     z: center.z + vecUp.z },
        { x: center.x + vecUp.x + vecRight.x * scale,y: center.y + vecUp.y + vecRight.y * scale,z: center.z + vecUp.z + vecRight.z * scale },
        // Middle row
        { x: center.x + vecLeft.x * scale,           y: center.y + vecLeft.y * scale,           z: center.z + vecLeft.z * scale },
        center,
        { x: center.x + vecRight.x * scale,          y: center.y + vecRight.y * scale,          z: center.z + vecRight.z * scale },
        // Bottom row
        { x: center.x + vecDown.x + vecLeft.x * scale, y: center.y + vecDown.y + vecLeft.y * scale, z: center.z + vecDown.z + vecLeft.z * scale },
        { x: center.x + vecDown.x,                       y: center.y + vecDown.y,                       z: center.z + vecDown.z },
        { x: center.x + vecDown.x + vecRight.x * scale,y: center.y + vecDown.y + vecRight.y * scale,z: center.z + vecDown.z + vecRight.z * scale }
    ];

    // 4. 3D頂点を遠近法を適用して2Dに投影
    const vertices2D = vertices3D.map(v => {
        const perspective = FOCAL_LENGTH / (FOCAL_LENGTH + v.z * Z_SCALE_FACTOR);
        return {
            x: v.x * canvas.width * perspective,
            y: v.y * canvas.height * perspective
        };
    });

    // 5. 描画する8つの三角形を定義 (頂点インデックスで指定)
    const triangles = [
        [0, 1, 3], [1, 4, 3], // Top-left quad
        [1, 2, 4], [2, 5, 4], // Top-right quad
        [3, 4, 6], [4, 7, 6], // Bottom-left quad
        [4, 5, 7], [5, 8, 7]  // Bottom-right quad
    ];

    // 6. 元画像（テクスチャ）上の対応する頂点を定義 (0-1の正規化座標)
    const uv_coords = [
        { u: 0, v: 0 }, { u: 0.5, v: 0 }, { u: 1, v: 0 },
        { u: 0, v: 0.5 }, { u: 0.5, v: 0.5 }, { u: 1, v: 0.5 },
        { u: 0, v: 1 }, { u: 0.5, v: 1 }, { u: 1, v: 1 }
    ];
    
    // 7. 各三角形を描画
    for (const tri of triangles) {
        const p0 = vertices2D[tri[0]], p1 = vertices2D[tri[1]], p2 = vertices2D[tri[2]];
        const uv0 = uv_coords[tri[0]], uv1 = uv_coords[tri[1]], uv2 = uv_coords[tri[2]];
        drawTexturedTriangle(p0, p1, p2, uv0, uv1, uv2);
    }
}

// --- 【新規】テクスチャマッピングされた三角形を描画するヘルパー関数 ---
function drawTexturedTriangle(p0, p1, p2, uv0, uv1, uv2) {
    ctx.save();

    // 1. 描画領域を三角形にクリッピング
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.closePath();
    ctx.clip();

    // 2. テクスチャ座標からピクセル座標へ変換
    const t0 = { x: uv0.u * muzzleImg.width, y: uv0.v * muzzleImg.height };
    const t1 = { x: uv1.u * muzzleImg.width, y: uv1.v * muzzleImg.height };
    const t2 = { x: uv2.u * muzzleImg.width, y: uv2.v * muzzleImg.height };

    // 3. アフィン変換行列を計算して、元画像の三角形を描画先の三角形にマッピングする
    // この計算は、2つの三角形間の変換を求める連立方程式を解いている
    const delta = t0.x * t1.y + t1.x * t2.y + t2.x * t0.y - t1.x * t0.y - t2.x * t1.y - t0.x * t2.y;
    if (Math.abs(delta) < 1e-6) { // 三角形が縮退している場合は描画しない
        ctx.restore();
        return;
    }

    const a = (p0.x * t1.y + p1.x * t2.y + p2.x * t0.y - p1.x * t0.y - p2.x * t1.y - p0.x * t2.y) / delta;
    const b = (p0.y * t1.y + p1.y * t2.y + p2.y * t0.y - p1.y * t0.y - p2.y * t1.y - p0.y * t2.y) / delta;
    const c = (p0.x * t2.x + p1.x * t0.x + p2.x * t1.x - p1.x * t2.x - p2.x * t0.x - p0.x * t1.x) / delta;
    const d = (p0.y * t2.x + p1.y * t0.x + p2.y * t1.x - p1.y * t2.x - p2.y * t0.x - p0.y * t1.x) / delta;
    const e = (p0.x * (t1.y * t2.x - t2.y * t1.x) + p1.x * (t2.y * t0.x - t0.y * t2.x) + p2.x * (t0.y * t1.x - t1.y * t0.x)) / delta;
    const f = (p0.y * (t1.y * t2.x - t2.y * t1.x) + p1.y * (t2.y * t0.x - t0.y * t2.x) + p2.y * (t0.y * t1.x - t1.y * t0.x)) / delta;

    // 4. 計算した行列を適用し、画像全体を描画する（クリッピング領域のみ表示される）
    ctx.transform(a, b, c, d, e, f);
    ctx.drawImage(muzzleImg, 0, 0);

    ctx.restore();
}

// --- UIリセット & ダウンロード設定 (変更なし) ---
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
