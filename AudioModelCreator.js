const createAndTrainBtn = document.querySelector("#createAndTrainBtn");
const testModelBtn = document.querySelector("#testModelBtn");
const loadModelBtn = document.querySelector("#loadModelBtn");
const micStartBtn = document.querySelector("#micStartBtn");
const micStopBtn = document.querySelector("#micStopBtn");
const recAudioBtn = document.querySelector("#recAudioBtn");

const audioPlayer = document.querySelector("#audioPlayer");

const moveBtn = document.querySelector("#move");
const moveBtn2 = document.querySelector("#move2");

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
const NUM_SECONDS = 3; // Número de segundos para el muestreo desde mic
const OVERLAP_FACTOR = 0.0; // Factor de superposición de los fotogramas

const CLASSES = ["background_noise", "crying_baby", "clock_alarm", "toilet_flush", "water_drops"];


function flattenQueue(queue) {
    const frameSize = queue[0].length;
    const data = new Float32Array(queue.length * frameSize);
    queue.forEach((d, i) => data.set(d, i * frameSize));
    return data;
}

let model;
let yamnet;

async function app() {
    let audioContext;
    let stream;

    const timeDataQueue = [];

    const trainDataArray = await (await fetch("data/trainData.json")).json();

    const testDataArray = await (await fetch("data/testData.json")).json();

    enableAllButtons(false);

    enableButton(moveBtn, true);
    enableButton(moveBtn2, true);
    yamnet = await loadYamnetModel();
    console.log("YamNet model loaded");
    enableButton(createAndTrainBtn, true);
    enableButton(loadModelBtn, true);
    enableButton(recAudioBtn, true);

    enableButton(testModelBtn, false);
    enableButton(micStartBtn, false);
    enableButton(micStopBtn, false);

    recAudioBtn.onclick = async () => recordAudio(5000);

    micStartBtn.onclick = async () => {

        stream = await getAudioStream();
        audioContext = new AudioContext({
            latencyHint: "playback",
            sampleRate: MODEL_SAMPLE_RATE
        });

        const streamSource = audioContext.createMediaStreamSource(stream);
        /*const processor = audioContext.createScriptProcessor(4096, 1, 1);
        streamSource.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = async (data) => {
            // console.log(data.inputBuffer.getChannelData(0));
            const inputBuffer = Array.from(await data.inputBuffer.getChannelData(0));
            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            const num_samples = timeDataQueue.length;
            if (num_samples >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
                const audioData = new Float32Array(timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS));
                 console.log(CLASSES[await predict(yamnet, model, audioData)]);
            }
        }*/
        await audioContext.audioWorklet.addModule("recorder.worklet.js");
        const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
        streamSource.connect(recorder).connect(audioContext.destination);

        enableButton(micStartBtn, false);
        enableButton(micStopBtn, true);

        recorder.port.onmessage =  async(e) => {
            const inputBuffer = Array.from(e.data);
            
            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            const num_samples = timeDataQueue.length;
            if (num_samples >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
                const audioData = new Float32Array(timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS));
                console.log(CLASSES[await predict(yamnet, model, audioData)]);
            }
        }
    };

    micStopBtn.onclick = () => {
        if (!Boolean(audioContext) || !Boolean(stream)) return; 
        audioContext.close();
        audioContext = null;
      
        timeDataQueue.splice(0);
        if (stream != null && stream.getTracks().length > 0) {
            stream.getTracks()[0].stop();
            enableButton(micStartBtn, true);
            enableButton(micStopBtn, false);
        }
    }


    loadModelBtn.onclick = async () => {
        model = await loadCustomAudioClassificationModelFromFile("./model/model.json");
        enableButton(testModelBtn, true);
        enableButton(micStartBtn, true);
        enableButton(createAndTrainBtn, false);
        enableButton(loadModelBtn, false);
    }

    testModelBtn.onclick = async () => {
        testCustomAudioClassificationModel(yamnet, model, testDataArray);
    };

    createAndTrainBtn.onclick = async () => {
        enableAllButtons(false);
        model = await createAndTrainCustomAudioClassificationModel(yamnet, trainDataArray);
        enableAllButtons(true);
        enableButton(createAndTrainBtn, false);
        enableButton(loadModelBtn, false);
    };

}

function recordAudio(millis) {
    const audioData = new Float32Array(Array.from(
        {
            length: MODEL_SAMPLE_RATE * millis / 1e3
        }, () => Math.random() * 2 - 1));

    const wavBytes = getWavBytes(audioData.buffer);

    const blob = new Blob([wavBytes], { 'type': 'audio/wav' });
    const audioURL = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = audioURL;
    a.download = "test.wav";
    a.click();
}

async function loadYamnetModel() {
    const model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
    return model;
}

async function testCustomAudioClassificationModel(yamnet, model, testDataArray) {
    const RANDOM = Math.floor((Math.random() * testDataArray.length));
    const testSample = testDataArray[RANDOM];
    // console.log(testSample);
    const audioData = await getTimeDomainDataFromFile(`data/testAudio/${testSample.fileName}`);
    playAudio(`data/testAudio/${testSample.fileName}`);
    const prediction = await predict(yamnet, model, audioData);
    console.log(CLASSES[prediction]);
}

async function predict(yamnet, model, audioData, vis = true) {
    const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
    // embeddings.print(true);
    const results = await model.predict(embeddings);
    results.print(true)
    const meanTensor = results.mean((axis = 0));
    // meanTensor.print();
    const argMaxTensor = meanTensor.argMax(0);

    if (vis) {
        const series = Array.from(audioData).map((y, x) => ({ x, y }));
        const audioDataSeries = {
            values: series,
            series: ['Waveform']
        }
     
        const audioDataSurface = {
            name: 'Waveform',
            tab: 'Charts'
        };
        tfvis.render.linechart(audioDataSurface, audioDataSeries, { zoomToFit: true });
     
        const predictionSeries = {
            values: results.arraySync(),
            xTickLabels: new Array(results.shape[0]).fill(0).map((e, i) => `${i + 1}`),
            yTickLabels: CLASSES,
     
     
        }
     
     
        const predictionSurface = tfvis.visor().surface({ name: 'Predictions', tab: 'Charts' });
        tfvis.render.heatmap(predictionSurface, predictionSeries);
     
     }
     

    embeddings.dispose();
    results.dispose();
    meanTensor.dispose();
    return argMaxTensor.dataSync()[0];
}

async function loadCustomAudioClassificationModelFromFile(url) {
    const model = await tf.loadLayersModel(url);
    model.summary();
    return model;
}

function logProgress(epoch, logs) {
    console.log(`Data for epoch ${epoch}, ${Math.sqrt(logs.loss)}`);
}



async function prepareTrainingData(yamnet, trainDataArray) {
    const INPUT_DATA = [];
    const OUTPUT_DATA = [];

    const context = new AudioContext({ latencyHint: "playback", sampleRate: MODEL_SAMPLE_RATE });
    for (let i = 0; i < trainDataArray.length; i++) {
        const audioData = await getTimeDomainDataFromFile(`data/trainAudio/${trainDataArray[i].fileName}`, context);
        const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
        const embeddingsArray = embeddings.arraySync();

        for (let j = 0; j < embeddingsArray.length; j++) {
            INPUT_DATA.push(embeddingsArray[j]);
            OUTPUT_DATA.push(trainDataArray[i].classNumber);
        }

    }

    tf.util.shuffleCombo(INPUT_DATA, OUTPUT_DATA);

    const inputTensor = tf.tensor2d(INPUT_DATA);
     inputTensor.print(true);

    const outputAsOneHotTensor = tf.oneHot(tf.tensor1d(OUTPUT_DATA, 'int32'), NUM_CLASSES);
    
    return [inputTensor, outputAsOneHotTensor];
}

async function createAndTrainCustomAudioClassificationModel(yamnet, trainDataArray) {

    const [inputTensor, outputAsOneHotTensor] = await prepareTrainingData(yamnet, trainDataArray);
    // outputAsOneHotTensor.print(true);

    const model = createModel();
    await trainModel(model, inputTensor, outputAsOneHotTensor);
    await saveModel(model);
    return model;
}

async function saveModel(model) {
    model.save('downloads://model');
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({ dtype: 'float32', inputShape: [INPUT_SHAPE], units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
    model.summary();
    return model;
}

async function trainModel(model, inputTensor, outputTensor) {
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const params = {
        shuffle: true,
        validationSplit: 0.15,
        batchSize: 16,
        epochs: 20,
        callbacks: [new tf.CustomCallback({ onEpochEnd: logProgress }),
        //tf.callbacks.earlyStopping({ monitor: 'loss', patience: 3 })
        ]
    };

    const results = await model.fit(inputTensor, outputTensor, params);
    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
    console.log("Average validation error loss: " +
        Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));
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

async function playAudio(url) {
    audioPlayer.src = url;
    audioPlayer.load();
    audioPlayer.onloadeddata = function () { audioPlayer.play(); };
}

async function getTimeDomainDataFromFile(url) {
    const audioContext = new AudioContext({
        sampleRate: MODEL_SAMPLE_RATE
    });
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
}

async function getEmbeddingsFromTimeDomainData(model, audioData) {
    const waveformTensor = tf.tensor(audioData);
    // waveformTensor.print(true);
    const [scores, embeddings, spectrogram] = model.predict(waveformTensor);
    waveformTensor.dispose();
    return embeddings;
}

function enableButton(buttonElement, enabled) {
    if (enabled) {
        buttonElement.removeAttribute("disabled");
    } else {
        buttonElement.setAttribute("disabled", "");
    }
}

function enableAllButtons(enable) {
    document.querySelectorAll("button").forEach(btn => {
        if (enable) {
            btn.removeAttribute("disabled");
        } else {
            btn.setAttribute("disabled", "");
        }
    });
}

app();

// Retorna Uint8Array de bytes WAV
function getWavBytes(buffer) {
    const numFrames = buffer.byteLength / Float32Array.BYTES_PER_ELEMENT;
    const headerBytes = getWavHeader(numFrames);
    const wavBytes = new Uint8Array(headerBytes.length + buffer.byteLength);

    // prepend header, then add pcmBytes
    wavBytes.set(headerBytes, 0);
    wavBytes.set(new Uint8Array(buffer), headerBytes.length);

    return wavBytes;
}

function getWavHeader(numFrames) {
    const numChannels = 1;
    const bytesPerSample = 4;

    const format = 3; //Float32

    const blockAlign = numChannels * bytesPerSample;
    const byteRate = MODEL_SAMPLE_RATE * blockAlign;
    const dataSize = numFrames * blockAlign;

    const buffer = new ArrayBuffer(44);
    const dv = new DataView(buffer);

    let p = 0;

    function writeString(s) {
        for (let i = 0; i < s.length; i++) {
            dv.setUint8(p + i, s.charCodeAt(i));
        }
        p += s.length;
    }

    function writeUint32(d) {
        dv.setUint32(p, d, true);
        p += 4;
    }

    function writeUint16(d) {
        dv.setUint16(p, d, true);
        p += 2;
    }

    writeString('RIFF');              // ChunkID
    writeUint32(dataSize + 36);       // ChunkSize
    writeString('WAVE');              // Format
    writeString('fmt ');              // Subchunk1ID
    writeUint32(16);                  // Subchunk1Size
    writeUint16(format);              // AudioFormat
    writeUint16(numChannels);         // NumChannels
    writeUint32(MODEL_SAMPLE_RATE);   // SampleRate
    writeUint32(byteRate);            // ByteRate
    writeUint16(blockAlign);          // BlockAlign
    writeUint16(bytesPerSample * 8);  // BitsPerSample
    writeString('data');              // Subchunk2ID
    writeUint32(dataSize);            // Subchunk2Size

    return new Uint8Array(buffer);
}


