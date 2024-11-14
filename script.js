let model;
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let isDrawing = false;

// 캔버스 초기화
function clearCanvas() {
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	document.getElementById('prediction').innerText = "Prediction: ";
}

// 드로잉 이벤트 설정
canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

function draw(event) {
	if (!isDrawing) return;
	ctx.lineWidth = 20;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'black';
	ctx.lineTo(event.offsetX, event.offsetY);
	ctx.stroke();
	ctx.beginPath();
	ctx.moveTo(event.offsetX, event.offsetY);
}

// OpenCV.js가 로드되고 준비가 되면 실행
function onOpenCVReady() {
	console.log('OpenCV.js is ready');
	// 여기서부터 OpenCV를 사용할 수 있습니다.
}

// 모델 로딩
async function loadModel() {
	model = await tf.loadLayersModel('http://localhost:1234/model.json');
	console.log("Model Loaded Successfully");
}

// 예측 기능
async function predict() {
    if (!model) await loadModel();

    // TensorFlow.js에서 CPU로 설정
    tf.setBackend('cpu');  // WebGL을 비활성화하고 CPU로 실행

    // 캔버스 이미지 가져오기
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // OpenCV.js로 이진화 처리
    let src = cv.matFromImageData(imageData);
    let gray = new cv.Mat();
    let thresholded = new cv.Mat();

    // 그레이스케일로 변환
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    // 이진화 처리 (배경을 흰색으로 처리)
    cv.threshold(gray, thresholded, 127, 255, cv.THRESH_BINARY_INV);

	// 잡음 제거
	//cv.medianBlur(thresholded, thresholded, 5);  // 5는 커널 크기

	// 엣지 검출 (선택적으로 사용)
	//cv.Canny(thresholded, thresholded, 50, 150);

    // 크기 조정 (28x28)
    let resized = new cv.Mat();
    cv.resize(thresholded, resized, new cv.Size(28, 28), 0, 0, cv.INTER_LINEAR);

    // 그레이스케일 이미지를 RGBA 이미지로 변환
    let rgba = new cv.Mat();
    cv.cvtColor(resized, rgba, cv.COLOR_GRAY2RGBA);

    // TensorFlow.js에서 사용할 수 있도록 ImageData로 변환
    let imageDataResized = new ImageData(new Uint8ClampedArray(rgba.data), rgba.cols, rgba.rows);

    // TensorFlow.js에 입력할 형태로 변환
    let tensor = tf.browser.fromPixels(imageDataResized, 1)  // 그레이스케일 이미지
        .toFloat()
        .expandDims(0)  // 배치 차원 추가 (배치 크기 1)
        .expandDims(-1) // 채널 차원 추가 (배치, 높이, 너비, 채널)
        .div(tf.scalar(255.0))  // 정규화: 0~255 -> 0~1로 변환
		.reshape([1, 784]);

    // 예측 수행
    const prediction = model.predict(tensor);
    const predictedValue = prediction.argMax(1).dataSync()[0];

    // 예측 결과 출력
	console.log("result : "+predictedValue)
    document.getElementById('prediction').innerText = `Prediction: ${predictedValue}`;

    // 전처리된 이미지를 캔버스에 출력
    let outputCanvas = document.getElementById('outputCanvas');
    let outputCtx = outputCanvas.getContext('2d');
    outputCanvas.width = resized.cols;
    outputCanvas.height = resized.rows;

    // 전처리된 이미지 그리기
    let outputImageData = new ImageData(new Uint8ClampedArray(rgba.data), rgba.cols, rgba.rows);
    outputCtx.putImageData(outputImageData, 0, 0);

    // 메모리 해제
    prediction.dispose();
    tensor.dispose();
    src.delete();
    gray.delete();
    thresholded.delete();
    resized.delete();
    rgba.delete();
}

// 버튼 이벤트 설정
document.getElementById('clearButton').addEventListener('click', clearCanvas);
document.getElementById('predictButton').addEventListener('click', predict);

// 페이지 로드 시 모델 로드
loadModel();

// OpenCV.js 로딩 후 초기화 호출
if (typeof cv !== 'undefined') {
	console.log('cv is loaded.');
	onOpenCVReady();
} else {
	document.addEventListener('opencvjsloaded', onOpenCVReady);
}

clearCanvas();