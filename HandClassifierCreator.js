const textMessage = document.querySelector("#message");
const createAndTrainBtn = document.querySelector("#createAndTrainBtn");
const testModelBtn = document.querySelector("#testModelBtn");
const loadModelBtn = document.querySelector("#loadModelBtn");

const gesture1Btn = document.querySelector("#gesture1Btn");
const gesture2Btn = document.querySelector("#gesture2Btn");
const gesture3Btn = document.querySelector("#gesture3Btn");
const gesture4Btn = document.querySelector("#gesture4Btn");
const moveBtn = document.querySelector("#move");
const moveBtn2 = document.querySelector("#move2");

const saveBtn = document.querySelector("#saveBtn");

const webcamEl = document.querySelector("#webcam");

const HAND_CLASSES = [ "Gesto 1", "Gesto 2", "Gesto 3", "Gesto 4"];
const NO_GESTURE_CLASS = "Ningún Gesto";
const CONFIDENCE_MINIMUM = 0.5


let classifier;
let keypoints;
let testModel = false;
let loadTestButtonWhenLoadingModel = false;

const totalElements = 21 * 3; // 21 filas y 3 columnas
const zerosArray = Array(totalElements).fill(0.0001);

async function app() {
    enableAllButtons(false);
    enableButton(moveBtn, true);
    enableButton(moveBtn2, true);
    // Load classifier
    initClassifier();

    //BORRAR
    enableButton(loadModelBtn, true);

    loadModelBtn.onclick = async () => {
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

    saveBtn.onclick = async () => {
        const json = await saveClassifier();
        descargarJSON(json, 'knnClassifier.json');
    };



    
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


    // Initialize UI components
    // (Assuming that enableAllButtons, enableButton, and other UI functions are defined elsewhere)

    enableButton(gesture1Btn, true)
    enableButton(gesture2Btn, true)
    enableButton(gesture3Btn, true)
    enableButton(gesture4Btn, true)
    enableButton(createAndTrainBtn, true);
    loadTestButtonWhenLoadingModel = true;
    if(classifier.getNumClasses() == HAND_CLASSES.length){
        enableButton(testModelBtn, true);
    }

    createAndTrainBtn.onclick = async () => {
        testModel = false;
        initClassifier();
    }

    

    testModelBtn.onclick = async () => {
        testModel = true;
    }

    

    gesture1Btn.onclick = async () => {
        addExample(HAND_CLASSES[0]);
    }

    gesture2Btn.onclick = async () => {
        addExample(HAND_CLASSES[1]);
    }

    gesture3Btn.onclick = async () => {
        addExample(HAND_CLASSES[2]);
    }

    gesture4Btn.onclick = async () => {
        addExample(HAND_CLASSES[3]);
    }


    // Handpose estimation loop
    async function handposeEstimation() {



        keypoints = null;
        let keypointsOld = null;
        while (true) {
            const img = await webcam.capture();
            const predictions = await handposeModel.estimateHands(img, {flipHorizontal: false});
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
                    
                    if(testModel){
                        
                        textMessage.textContent = "Gesto detectado: " + classifyGesture(keypoints);
                    }




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

function initClassifier(){
    testModel = false;
    classifier = knnClassifier.create();
    textMessage.textContent = "";
    //classifier.addExample(tf.tensor(zerosArray, [21, 3]), NO_GESTURE_CLASS);
    enableButton(testModelBtn,false);
    enableButton(saveBtn,false);
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

function clearCanvas(canvas, context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
}


//////////////////////////////////////////////////////////////////////////////////////////////




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


async function addExample(label) {
    if(keypoints !== null){
        classifier.addExample(tf.tensor(keypoints), label);
        //classifier.addExample(tf.tensor(zerosArray, [21, 3]), NO_GESTURE_CLASS);
        if(classifier.getNumClasses() == HAND_CLASSES.length){
            enableAllButtons(true);
        }
    }
}


async function saveClassifier() {
    const dataset = classifier.getClassifierDataset();
    var datasetObj = {};
    Object.keys(dataset).forEach((key) => {
        let tensor = dataset[key];
        let data = tensor.dataSync();
        datasetObj[key] = {
            data: Array.from(data),
            shape: tensor.shape
        };
    });
    const jsonStr = JSON.stringify(datasetObj);
    // Guardar jsonStr según lo necesites
    return jsonStr;
}



async function loadClassifier(jsonStr) {
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
    console.log("joder funciona")
    if(loadTestButtonWhenLoadingModel){
        enableButton(testModelBtn, true);
    }
    enableButton(saveBtn, true);
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


function descargarJSON(jsonData, nombreArchivo) {
    const jsonString = JSON.stringify(jsonData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const elemento = document.createElement('a');
    elemento.href = url;
    elemento.download = nombreArchivo;
    elemento.style.display = 'none';
    document.body.appendChild(elemento);

    elemento.click();

    document.body.removeChild(elemento);
    URL.revokeObjectURL(url);
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





