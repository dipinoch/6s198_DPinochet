
import * as tf from '@tensorflow/tfjs';
import { Data } from './data';
import { isInt } from './node_modules/@tensorflow/tfjs-core/dist/util';
import * as ui from './ui';
import fashion_mnist from './fashion_mnist_images.png';


const xhrDatasetConfigs = {
  MNIST: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
      dataType: 'png',
      shape: [28, 28, 1]
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
      label_id : null,
    }],
  },

  Fashion_MNIST: {
    data: [{
      name: 'images',
      // path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
      path: fashion_mnist,
      dataType: 'png',
      shape: [28, 28, 1]
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
      label_id: {0:'T-shirt/Top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandel', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot' },
    }],
  },

  CIFAR_10: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/cifar10_images.png',
      dataType: 'png',
      shape: [32, 32, 3]
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/cifar10_labels_uint8',
      dataType: 'uint8',
      shape: [10],
      label_id : {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"},
    }],
    },
}

// To change between MNIST and Fashion_MNIST, change the definitions
// of datasetName by commenting and uncommenting the
// lines below, and reload the page

const datasetName = 'MNIST';
//const datasetName = 'Fashion_MNIST';
//const datasetName = 'CIFAR_10';



const image_shape = xhrDatasetConfigs[datasetName]['data'][0]['shape'];

const TRAIN_TEST_RATIO = 5 / 6;

let data;
let num_images;
async function loadData() {
  const image_path = xhrDatasetConfigs[datasetName]['data'][0]['path'];
  const lables_path = xhrDatasetConfigs[datasetName]['data'][1]['path'];
  const labels_shape = xhrDatasetConfigs[datasetName]['data'][1]['shape'];
  data = new Data(image_path, lables_path, image_shape, labels_shape, TRAIN_TEST_RATIO);
  await data.load();
  num_images = data.datasetImages.length/image_shape.reduce((a, b) => a * b);
}

/**
 * ===============================
 * HYPERPARAMETERS
 * ===============================
 */

const BATCH_SIZE =64;

const NUM_BATCHES =200;

// flag to control whether or not to do testing
const DO_TESTING = true;
// The number of test examples to predict each time we test. Because we don't
// update model weights during testing this value doesn't affect model training.
const TEST_BATCH_SIZE = 60;
// The number of training batches we will run between each test batch.
const TEST_ITERATION_FREQUENCY = 5;

/**
 * ===============================
 * CONSTRUCT YOUR MODEL HERE
 * ===============================
 */
// Create a sequential neural network model. tf.sequential provides an API for
// creating "stacked" models where the output from one layer is used as the
// input to the next layer.
const model = tf.sequential();

// First flatten the image
// First layer must have an input shape defined.
model.add(tf.layers.flatten({
  inputShape: image_shape
}))

// Add a fully conected (dense) layer 
model.add(tf.layers.dense({
  units: 100, 
  kernelInitializer: 'varianceScaling'
}));

//Add a Relu 
model.add(tf.layers.dense({
    units:10,
    activation:'relu',
    kernelInitializer: 'varianceScaling'
    }));

//Add a fully conected (dense) layer 
model.add(tf.layers.dense({
     units: 10, 
      kernelInitializer: 'varianceScaling'
    }));
    
//Add a Relu 
model.add(tf.layers.dense({
    units:10,
    activation:'relu',
    kernelInitializer: 'varianceScaling'
    }));



//Add softmax
model.add(tf.layers.softmax());

model.compile({
  optimizer: 'sgd', 
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});


async function train() {
  ui.isTraining();

  // We'll keep a buffer of loss and accuracy values over time.
  const lossValues = [];
  const accuracyValues = [];

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < NUM_BATCHES; i++) {
    const [batch, validationData] = tf.tidy(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      batch.xs = batch.xs.reshape([BATCH_SIZE].concat(image_shape));

      let validationData;
      // Every few batches test the accuracy of the model.
      if (i % TEST_ITERATION_FREQUENCY === 0 && DO_TESTING) {
        const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
        validationData = [
          // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
          // that we can feed it to our convolutional neural net.
          testBatch.xs.reshape([TEST_BATCH_SIZE].concat(image_shape)), testBatch.labels
        ];
      }
      return [batch, validationData];
    });

    // The entire dataset doesn't fit into memory so we call train repeatedly
    // with batches using the fit() method.
    const history = await model.fit(
        batch.xs, batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    // Plot loss / accuracy.
    lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});
    ui.plotLosses(lossValues);

    if (validationData != null) {
      accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
      ui.plotAccuracies(accuracyValues);
    }

    // Call dispose on the training/test tensors to free their GPU memory.
    tf.dispose([batch, validationData]);

    // tf.nextFrame() returns a promise that resolves at the next call to
    // requestAnimationFrame(). By awaiting this promise we keep our model
    // training from blocking the main UI thread and freezing the browser.
    await tf.nextFrame();
  }
}

async function showPredictions() {
  const testExamples = 50;
  const batch = data.nextTotalBatch(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1].concat(image_shape)));
    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());
    const label_id = xhrDatasetConfigs[datasetName]['data'][1]['label_id']
    ui.showTestResults(batch, predictions, labels, image_shape, label_id);
  });
}

async function identifyImage(){
  const n = document.getElementById("imageNumber").value;
  if (n>=0 && n< num_images && isInt(n)){
    const image = await data.getNthImage(n);
    const imageTensor = tf.tensor4d(image, [1].concat(image_shape));
    const classes = await model.predict(imageTensor);
    ui.showimage(image, image_shape);
    ui.showResults(await classes.data(), xhrDatasetConfigs[datasetName]['data'][1]['label_id']);
  }else{
    throw "Input must be an integer between 0 and " + (num_images-1) + " inclusive.";
  }
}

async function setupdoc(){
  document.getElementById("imageIdentifier").innerText = "To see your model identify an image, enter an integer and press submit:"; 
  const input = document.createElement('input');
  input.id = "imageNumber";
  input.value = 0;
  input.min = 0;
  input.max = num_images-1;
  document.getElementById("imageIdentifier").appendChild(input)
  const submit = document.createElement('button');
  submit.innerText = "Submit";
  submit.id = "submit";
  submit.onclick = await identifyImage;;
  document.getElementById("imageIdentifier").appendChild(submit);
}
// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
async function init() {
  document.getElementById('title').innerText = 'TensorFlow.js: Train ' + datasetName + ' with the Layers API.'
  await loadData();
  await train();
  if (DO_TESTING) {
    await setupdoc()
    showPredictions()
  };
}
init();
