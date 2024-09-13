import * as tf from "@tensorflow/tfjs";
import * as tfd from "@tensorflow/tfjs-data";

// registering service worker:
navigator.serviceWorker.register("/sw.js");

const recordButtons = document.getElementsByClassName("record-button");
const buttonsContainer = document.getElementById("buttons-container");

const trainButton = document.getElementById("train");
const predictButton = document.getElementById("predict");
const statusElement = document.getElementById("status");

let webcam, initialModel, mouseDown, newModel;

// const totals = [0, 0];
// const labels = ["left", "right"];
const learningRate = 0.0001;
const batchSizeFraction = 0.4;
const epochs = 30;
const denseUnits = 100;

let isTraining = false;
let isPredicting = false;

let labels = [];
let totals = [];

const videos = {
  happy: [
    "https://www.youtube.com/embed/LjhCEhWiKXk",

    "https://www.youtube.com/embed/Y66j_BUCBMY",

    "https://www.youtube.com/embed/iPUmE-tne5U",

    "https://www.youtube.com/embed/nfWlot6h_JM",

    "https://www.youtube.com/embed/wsdy_rct6uo",

    "https://www.youtube.com/embed/ru0K8uYEZWw",

    "https://www.youtube.com/embed/Pw-0pbY9JeU",

    "https://www.youtube.com/embed/hT_nvWreIhg",
    "https://www.youtube.com/embed/HCjNJDNzw8Y",
  ],

  sad: [
    "https://www.youtube.com/embed/YQHsXMglC9A",
    "https://www.youtube.com/embed/hLQl3WQQoQ0",
    "https://www.youtube.com/embed/RBumgq5yVrA",
    "https://www.youtube.com/embed/6EEW-9NDM5k",
    "https://www.youtube.com/embed/0G3_kG5FFfQ",

    "https://www.youtube.com/embed/VT1-sitWRtY",

    "https://www.youtube.com/embed/HLphrgQFHUQ",

    "https://www.youtube.com/embed/koJlIGDImiU",

    "https://www.youtube.com/embed/My2FRPA3Gf8",
  ],
};

document.getElementById("add-label").onclick = () => {
  const labelInput = document.getElementById("label-input");
  const label = labelInput.value.trim();

  if (label && !labels.includes(label)) {
    labels.push(label);
    totals.push(0);

    const button = document.createElement("button");
    button.classList.add("record-button");
    button.innerText = `Add ${label} sample`;
    button.onclick = () => handleAddExample(labels.indexOf(label));

    const total = document.createElement("p");
    total.innerHTML = `<span id="${label}-total">0</span> examples`;

    buttonsContainer.appendChild(button);
    buttonsContainer.appendChild(total);

    labelInput.value = ""; // Clear input field
  }
};

const loadModel = async () => {
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );

  const layer = mobilenet.getLayer("conv_pw_13_relu");
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
};

const init = async () => {
  webcam = await tfd.webcam(document.getElementById("webcam"));

  initialModel = await loadModel();
  statusElement.style.display = "none";
  // document.getElementById("controller").style.display = "block";
};

init();

buttonsContainer.onmousedown = (e) => {
  const buttonIndex = Array.from(recordButtons).indexOf(e.target);
  if (buttonIndex !== -1) {
    handleAddExample(buttonIndex);
  }
};

buttonsContainer.onmouseup = () => {
  mouseDown = false;
};

const handleAddExample = async (labelIndex) => {
  if (labels.length < 2) {
    alert("Please add at least two labels before adding samples.");
    return;
  }
  mouseDown = true;
  const total = document.getElementById(labels[labelIndex] + "-total");

  while (mouseDown) {
    addExample(labelIndex);
    total.innerText = ++totals[labelIndex];

    await tf.nextFrame();
  }

  trainButton.style.display = "inline";
};

let xs, xy;

const addExample = async (index) => {
  let img = await getImage();
  let example = initialModel.predict(img);

  const y = tf.tidy(() =>
    tf.oneHot(tf.tensor1d([index]).toInt(), labels.length)
  );

  if (xs == null) {
    xs = tf.keep(example);
    xy = tf.keep(y);
  } else {
    const previousX = xs;
    xs = tf.keep(previousX.concat(example, 0));

    const previousY = xy;
    xy = tf.keep(previousY.concat(y, 0));

    previousX.dispose();
    previousY.dispose();
    y.dispose();
    img.dispose();
  }
};

const getImage = async () => {
  const img = await webcam.capture();
  const processedImg = tf.tidy(() =>
    img.expandDims(0).toFloat().div(127).sub(1)
  );
  img.dispose();

  return processedImg;
};

trainButton.onclick = async () => {
  train();

  statusElement.style.display = "block";
  statusElement.innerHTML = "Training...";

  await newModel.save("indexeddb://my-trained-model");
  statusElement.innerHTML = "Model trained and saved to IndexedDB!";

  localStorage.setItem("modelLabels", JSON.stringify(labels));
};

const loadTrainedModel = async () => {
  try {
    newModel = await tf.loadLayersModel("indexeddb://my-trained-model");
    statusElement.innerHTML = "Model loaded from IndexedDB!";

    // Load the labels from localStorage
    const storedLabels = localStorage.getItem("modelLabels");
    if (storedLabels) {
      labels = JSON.parse(storedLabels);
    }

    // Optionally, restore the totals array
    totals = labels.map(() => 0); // Initialize totals to zero or any other logic
  } catch (error) {
    statusElement.innerHTML = "Failed to load model from IndexedDB.";
    console.error("Error loading model: ", error);
  }
};

// Call this function to load the model when needed
loadTrainedModel();

const train = () => {
  isTraining = true;
  if (!xs) {
    throw new Error("You forgot to add examples before training");
  }

  newModel = tf.sequential({
    layers: [
      tf.layers.flatten({
        inputShape: initialModel.outputs[0].shape.slice(1),
      }),
      tf.layers.dense({
        units: denseUnits,
        activation: "relu",
        kernelInitializer: "varianceScaling",
        useBias: true,
      }),
      tf.layers.dense({
        units: labels.length,
        kernelInitializer: "varianceScaling",
        useBias: true,
        activation: "softmax",
      }),
    ],
  });

  const optimizer = tf.train.adam(learningRate);
  newModel.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });

  const batchSize = Math.floor(xs.shape[0] * batchSizeFraction);

  newModel.fit(xs, xy, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        statusElement.innerHTML = "Loss: " + logs.loss.toFixed(5);
      },
    },
  });

  isTraining = false;
};

// predictButton.onclick = async () => {
//   isPredicting = true;
//   while (isPredicting) {
//     const img = await getImage();

//     const initialModelPrediction = initialModel.predict(img);
//     const predictions = newModel.predict(initialModelPrediction);

//     const predictedClass = predictions.as1D().argMax();
//     const classId = (await predictedClass.data())[0];
//     console.log(labels[classId]);

//     img.dispose();
//     await tf.nextFrame();
//   }
// };
// document.getElementById("take-photo").onclick = async () => {
//   const imgElement = document.getElementById("captured-photo");

//   // Capture the current image from the webcam
//   let imgTensor = await getImage();

//   imgTensor = tf.squeeze(imgTensor, 0);

//   // Rescale the tensor values to be within [0, 1]
//   imgTensor = tf.tidy(() => {
//     return imgTensor.add(1).div(2); // Convert from range [-1, 1] to [0, 1]
//   });

//   console.log(imgTensor.shape);

//   // Convert the image tensor to a data URL and display it
//   const dataUrl = await tf.browser.toPixels(imgTensor, imgElement);

//   // Dispose of the tensor as it's no longer needed
//   imgTensor.dispose();

//   // Predict the label of the captured image
//   const prediction = await predictCapturedPhoto();
//   console.log("Predicted Label: ", prediction);
// };

// const predictCapturedPhoto = async () => {
//   const imgElement = document.getElementById("captured-photo");

//   // Convert the displayed image back to a tensor
//   const imgTensor = tf.browser.fromPixels(imgElement);

//   // Process the image tensor
//   const processedImg = tf.tidy(() =>
//     imgTensor.expandDims(0).toFloat().div(127).sub(1)
//   );

//   // Predict the class using the trained model
//   const initialModelPrediction = initialModel.predict(processedImg);
//   const predictions = newModel.predict(initialModelPrediction);

//   const predictedClass = predictions.as1D().argMax();
//   const classId = (await predictedClass.data())[0];

//   // Dispose of the tensor to free memory
//   imgTensor.dispose();
//   processedImg.dispose();

//   return labels[classId];
// };

// document.getElementById("take-photo").onclick = async () => {
//   const imgElement = document.getElementById("captured-photo");

//   // Capture the current image from the webcam
//   let imgTensor = await getImage();

//   // imgTensor is now [1, 224, 224, 3], remove the first dimension
//   // using tf.squeeze to get a tensor of shape [224, 224, 3]

//   imgTensor = tf.squeeze(imgTensor, 0);

//   // Rescale the tensor values to be within [0, 1]
//   imgTensor = tf.tidy(() => {
//     return imgTensor.clipByValue(0, 1); // Clamp values between 0 and 1
//   });

//   console.log(imgTensor.shape);

//   // Convert the image tensor to a data URL and display it
//   await tf.browser.toPixels(imgTensor, imgElement);

//   // Dispose of the tensor as it's no longer needed
//   imgTensor.dispose();

//   // Predict the label of the captured image
//   const prediction = await predictCapturedPhoto();
//   console.log("Predicted Label: ", prediction);
// };

document.getElementById("take-photo").onclick = async () => {
  const imgElement = document.getElementById("captured-photo");

  // Capture the current image from the webcam
  let imgTensor = await getImage();

  // Ensure tensor is [height, width, channels]
  imgTensor = tf.squeeze(imgTensor, 0);

  // Rescale the tensor values to be within [0, 1]
  imgTensor = tf.tidy(() => imgTensor.clipByValue(0, 1));

  const canvas = document.getElementById("captured-photo");

  // Convert the image tensor to a data URL and display it
  await tf.browser.toPixels(imgTensor, canvas);

  // Dispose of the tensor as it's no longer needed
  imgTensor.dispose();

  // Predict the label of the captured image
  const prediction = await predictCapturedPhoto();
  console.log("Predicted Label: ", prediction);
};

const predictCapturedPhoto = async () => {
  const imgElement = document.getElementById("captured-photo");

  // Convert the displayed image back to a tensor
  let imgTensor = tf.browser.fromPixels(imgElement);

  // Process the image tensor and ensure values are in range
  const processedImg = tf.tidy(
    () => imgTensor.expandDims(0).toFloat().div(127).sub(1).clipByValue(-1, 1) // Clamp between [-1, 1]
  );

  // Predict the class using the trained model
  const initialModelPrediction = initialModel.predict(processedImg);
  const predictions = newModel.predict(initialModelPrediction);

  const predictedClass = predictions.as1D().argMax();
  const classId = (await predictedClass.data())[0];

  // Dispose of the tensor to free memory
  imgTensor.dispose();
  processedImg.dispose();

  const predictedLabel = labels[classId];
  console.log("Predicted Label: ", predictedLabel);

  // Update the emoji container based on the prediction
  const emojiContainer = document.getElementById("emoji-container");
  if (predictedLabel === "happy") {
    emojiContainer.innerText = "😊"; // Happy emoji
  } else if (predictedLabel === "sad") {
    emojiContainer.innerText = "😢"; // Sad emoji
  }

  // Announce the detected emotion with a female voice

  const utterance = new SpeechSynthesisUtterance(
    predictedLabel === "happy" || predictedLabel === "sad"
      ? `Hey ${predictedLabel} folk, feel these videos!:`
      : `Hey ${predictedLabel} folk, you get videos only for happy and sad's emotion.`
  );

  // Function to set the voice
  const setVoice = () => {
    const voices = window.speechSynthesis.getVoices();
    const femaleVoice = voices.find(
      (voice) =>
        voice.name.includes("Female") ||
        voice.name.includes("Google UK English Female") ||
        voice.name.includes("Samantha") ||
        voice.gender === "female"
    );

    if (femaleVoice) {
      utterance.voice = femaleVoice;
    } else {
      console.log("No female voice found, using the default voice.");
    }

    window.speechSynthesis.speak(utterance);
  };

  // Ensure voices are loaded before setting the voice
  if (window.speechSynthesis.getVoices().length === 0) {
    window.speechSynthesis.onvoiceschanged = setVoice;
  } else {
    setVoice();
  }

  // Shuffle the array and select the first 3 videos
  const shuffledVideos = videos[predictedLabel].sort(() => 0.5 - Math.random());
  const selectedVideos = shuffledVideos.slice(0, 3);

  // Clear previous iframes
  if (predictedLabel === "happy" || predictedLabel === "sad") {
    const parentIframe = document.getElementById("iframe");
    parentIframe.innerHTML = ""; // Clear any previous videos

    // Embed the selected videos
    selectedVideos.forEach((videoUrl) => {
      const iframe = document.createElement("iframe");
      iframe.src = videoUrl;
      iframe.width = "450";
      iframe.height = "315";
      iframe.allow =
        "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
      iframe.allowFullscreen = true;
      iframe.style.display = "block";
      iframe.style.marginBottom = "10px"; // Optional: add some spacing between iframes

      // Fallback link in case the video cannot be embedded
      const fallbackLink = document.createElement("a");
      fallbackLink.href = videoUrl;
      fallbackLink.target = "_blank"; // Open in a new tab
      fallbackLink.innerText = "Watch Video";
      fallbackLink.style.display = "none";

      // Append the iframe and fallback link
      parentIframe.appendChild(iframe);
      parentIframe.appendChild(fallbackLink);

      // Add an event listener to handle loading errors
      iframe.onerror = () => {
        iframe.style.display = "none"; // Hide the iframe
        fallbackLink.style.display = "block"; // Show the link instead
      };
    });
  }

  return predictedLabel;
};

// for web storage:

(async function () {
  if (navigator.storage && navigator.storage.persist) {
    if (!(await navigator.storage.persisted())) {
      const result = await navigator.storage.persist();
      console.log(`Was Persistent Storage Request granted? ${result}`);
    } else {
      console.log(`Persistent Storage already granted`);
    }
  }
})();

(async function () {
  if (navigator.storage && navigator.storage.estimate) {
    const q = await navigator.storage.estimate();
    console.log(`quota available: ${parseInt(q.quota / 1024 / 1024)}MiB`);
    console.log(`quota usage: ${q.usage / 1024}KiB`);
  }
})();
