const textMessage = document.querySelector("#message");

const audioPlayer = document.querySelector("#audioPlayer");

const webcamEl = document.querySelector("#webcam");


/* El código está escrito así con fines educativos. 
 * No es el código que usaríamos en producción
 */
const MODEL_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";

/* 
 * Parámetros para la creación del modelo
 */
const INPUT_SHAPE = 1024;
const NUM_CLASSES = 5;

/* 
 * Parámetros para el procesamiento de audio
 */
const MODEL_SAMPLE_RATE = 16000; // Frecuencia de muestreo para YAMNet
const NUM_SECONDS = 1; // Número de segundos para el muestreo desde mic
const OVERLAP_FACTOR = 0.0; // Factor de superposición de los fotogramas


const CLASSES = ["background_noise", "crying_baby", "clock_alarm", "toilet_flush", "water_drops"];
const HAND_CLASSES = [ "Gesto 1", "Gesto 2", "Gesto 3", "Gesto 4"];
const NO_GESTURE_CLASS = "Ningún Gesto";
const CONFIDENCE_MINIMUM = 0.5

function flattenQueue(queue) {
    const frameSize = queue[0].length;
    const data = new Float32Array(queue.length * frameSize);
    queue.forEach((d, i) => data.set(d, i * frameSize));
    return data;
}

let model;
let yamnet;
let streamAudio;
let audioTextReset = 10;
let audioTextChange = 1;
let audioTextCounter = 0;
let classifier = knnClassifier.create();
let poseClockBufferWait = 60;
let poseClockBuffer = 0;

let poseClockCounter = 0;
let poseClockStart = 10;
let gesture = NO_GESTURE_CLASS;

async function app() {
    // Load YAMNet model
    (async () => {
    console.log("YamNet model loading");
    yamnet = await loadYamnetModel();
    model = await loadCustomAudioClassificationModelFromFile("./model/model.json");
    console.log("YamNet model loaded");
    
    streamAudio = await getAudioStream();
    if (streamAudio) {
        processAudio(streamAudio);
    }

    console.log(streamAudio);
    }) ();
    

    // Load classifier
    initClassifier();


    // Load handpose model and set up webcam
    const handposeModel = await handpose.load();
    const webcam = await tf.data.webcam(webcamEl, {
        resizeWidth: 256,
        resizeHeight: 256
    });

    const camerabbox = webcamEl.getBoundingClientRect();
    canvas.style.top = camerabbox.y + "px";
    canvas.style.left = camerabbox.x + "px";
    const context = canvas.getContext("2d");
    context.translate(webcamEl.width, 0);
    context.scale(-1, 1);



    // Handpose estimation loop
    async function handposeEstimation() {
        keypoints = null;
        let keypointsOld = null;
        let lastGesture = gesture;
        while (true) {
            const img = await webcam.capture();
            const predictions = await handposeModel.estimateHands(img, {flipHorizontal: false});
            
            //Reinicia la cuenta de frames si se detecta un gesto distinto al anterior
            if(lastGesture.localeCompare(gesture) != 0){
                poseClockCounter= 0;
            }
            poseClockBuffer++;
            lastGesture = gesture;

            if (predictions.length > 0) {
                console.log(predictions);
                for (let i = 0; i < predictions.length; i++) {
                    keypointsOld = keypoints;
                    keypoints = predictions[i].landmarks;
                    if(keypoints == null || arraysEqual(keypoints, keypointsOld)){
                        keypoints = null;
                        clearCanvas(canvas, context);
                    }
                    context.clearRect(0, 0, canvas.width, canvas.height);
                    for (let i = 0; i < keypoints.length; i++) {
                        const [x, y, z] = keypoints[i];
                        drawCircle(context, x, y, 3, "#003300");
                    }

                    gesture = await classifyGesture(keypoints);

                    textMessage.textContent = "Gesto detectado: " + gesture;
                    handleGestureEvent(gesture);

                }
            } else{
                keypoints = null;
                clearCanvas(canvas, context);
            }
            img.dispose();
            await tf.nextFrame();
        }
    }


    // Start the handpose estimation in a separate async function
    handposeEstimation();

}

// Start the app

app();

async function predict(yamnet, model, audioData) {
    const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
    // embeddings.print(true);
    const results = await model.predict(embeddings);
    results.print(true)
    const meanTensor = results.mean((axis = 0));
    // meanTensor.print();
    const argMaxTensor = meanTensor.argMax(0);     

    embeddings.dispose();
    results.dispose();
    meanTensor.dispose();
    return argMaxTensor.dataSync()[0];
}

async function predictFromModel(audioData){
    return await predict(yamnet, model, audioData);
}

async function loadYamnetModel() {
    const model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
    return model;
}

async function getAudioStream(audioTrackConstraints) {
    let options = audioTrackConstraints || {};
    try {
        return await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: {
                sampleRate: options.sampleRate || MODEL_SAMPLE_RATE,
                sampleSize: options.sampleSize || 16,
                channelCount: options.channelCount || 1
            }
        });
    } catch (e) {
        console.error(e);
        return null;
    }
}


async function processAudio(stream) {
    const audioContext = new AudioContext({
        latencyHint: 'playback',
        sampleRate: MODEL_SAMPLE_RATE
    });
    const streamSource = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(16384, 1, 1);
    streamSource.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = async (event) => {
        const inputBuffer = event.inputBuffer.getChannelData(0);
        const prediction = await predictFromModel(inputBuffer);
        handleSoundEvent(prediction);
    };
}


function handleSoundEvent(prediction) {
    let text = null;
    switch(prediction){
        case 1:
            text = "Hay un bebé llorando, ¿es tuyo?👶🏻"
            break;
        case 2:
            text = "¡Atento, hay una alarma sonando!⏰"
            break;
        case 3:
            text = "¿Has tirado tú de la cadena?🚽"
            break;
        case 4:
            text = "Algo está goteando, ¿has cerrado bien el grifo?💧"
            break;
    }
    audioTextCounter++;
    if(prediction != 0){
        if(audioTextCounter >= audioTextChange){
            textMessage.textContent = text;
        }
        audioTextCounter = 0;
        //console.log("Evento de sonido detectado:", CLASSES[prediction]);
    }
    else{    
        if(audioTextCounter >= audioTextReset){
            audioTextCounter = 0;
            textMessage.textContent = "";
        }
    }
}

async function getEmbeddingsFromTimeDomainData(model, audioData) {
    //console.log(audioData);
    const waveformTensor = tf.tensor(audioData);
    // waveformTensor.print(true);
    const [scores, embeddings, spectrogram] = model.predict(waveformTensor);
    waveformTensor.dispose();
    return embeddings;
}

async function loadCustomAudioClassificationModelFromFile(url) {
    const model = await tf.loadLayersModel(url);
    model.summary();
    return model;
}

function drawCircle(context, cx, cy, radius, color) {
    context.beginPath();
    context.arc(cx, cy, radius, 0, 2 * Math.PI, false);
    context.fillStyle = "red";
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = color;
    context.stroke();
}

function arraysEqual(arr1, arr2) {
    if (arr2 == null) return false;
    if (arr1.length !== arr2.length) return false;
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) return false;
    }
    console.log("COINCIDEN");
    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////


function initClassifier(){
    fetch('./knn/knnClassifier.json')
    .then(response => {
        // Asegúrate de que la respuesta esté en formato JSON.
        //console.log(response);
        return response.json();
    })
    .then(data => {
        // Aquí 'data' es el contenido de tu archivo JSON.
        //console.log(data);
        loadClassifier(data);
    })
    .catch(error => {
        // Manejo de errores en caso de que algo salga mal.
        console.error('Error al leer el archivo JSON:', error);
    });
}




function loadClassifier(jsonStr) {
    const datasetObj = JSON.parse(jsonStr);
    const dataset = {};
    Object.keys(datasetObj).forEach((key) => {
        let entry = datasetObj[key];
        let tensor = tf.tensor(entry.data, entry.shape);
        dataset[key] = tensor;
    });
    classifier.setClassifierDataset(dataset);

    if(classifier.getNumClasses() < HAND_CLASSES){
        throw Error("NUMERO DE CLASES NO VALIDO");
    }
    console.log(classifier)
}



async function classifyGesture(keypoints) {
    let res = NO_GESTURE_CLASS;
    if(keypoints == null || classifier.getNumClasses() < 4) {
        return res;
    }
    const result = await classifier.predictClass(tf.tensor(keypoints), HAND_CLASSES.length); // k = 3
    console.log(result); // Muestra el resultado en la consola o actualiza la UI
    
    for (const [key, confidence] of Object.entries(result.confidences)) {
        if(confidence >= CONFIDENCE_MINIMUM) {
            segura = true;
            res = key;
            break; // Rompe el bucle si encuentra una clase con confianza suficiente
        }
    }

    console.log(res);
    return res;
    
}




function clearCanvas(canvas, context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
}


function handleGestureEvent(gesture){
    // No activa acción hasta que se llega a un número de frames desde la anterior acción
    poseClockCounter++;
    if(poseClockBuffer < poseClockBufferWait || poseClockCounter < poseClockStart){
        return;
    }
    poseClockBuffer = 0;
    poseClockCounter = 0;
    var currentZoom = document.body.style.zoom || 100;
    currentZoom = parseInt(currentZoom);
    switch(gesture){
        case "Gesto 1":
            document.body.style.zoom = currentZoom + 10 + "%";
            break;
        case "Gesto 2":
            document.body.style.zoom = currentZoom - 10 + "%";
            break;
        case "Gesto 3":
            document.body.style.color = "yellow";
            break;
        case "Gesto 4":
            document.body.style.color = "white";
            break;
    }
}