import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node-gpu';
tf.setBackend('tensorflow');

import * as fs from 'fs';
import * as timer from 'node-simple-timer';
import DataHelper from '../../helpers/DataHelper';

async function main() {
    let model: tf.Sequential = initModel();
    // Download if have no directory
    if (!fs.existsSync('./db/mnist-db'))
        await DataHelper.downloadFilesFromUrls([
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ], './db/mnist-db');
    // // Then load each of them into an array of numerics
    // Get in total 10,000 images and labels for training
    let trainImages = await loadImages('./db/mnist-db/train-images-idx3-ubyte', 10000 * 784);
    let trainLabels = await loadLabels('./db/mnist-db/train-labels-idx1-ubyte', 10000 * 1);
    // Get in total 5,000 images and labels for testing
    let testImages = await loadImages('./db/mnist-db/t10k-images-idx3-ubyte', 5000 * 784);
    let testLabels = await loadLabels('./db/mnist-db/t10k-labels-idx1-ubyte', 5000 * 1);
    // We break down dataset into individual sets of 200 images and labels for each training
    // This will drastically reduce time since the computation has to look for leaser data each iteration
    // This will also reduce overfitting
    let step = 0;
    let batchSize = 200;
    let numRetries = 5;
    // Also measure timming performance
    let totalTimer = new timer.Timer();
    totalTimer.start();
    while (trainImages.length) {
        // // Prepare data before training
        let xcur = trainImages.splice(0, batchSize * 784);
        let ycur = trainLabels.splice(0, batchSize * 1);
        let xs = tf.tensor(xcur).reshape([batchSize, 28, 28, 1]);
        let ys = tf.oneHot(tf.tensor1d(ycur, 'int32'), 10).toFloat();
        // Lets retry learning with the same data patch
        for (let trial = 0; trial < numRetries; trial++) {
            let history: any = await model.fit(xs, ys, {
                shuffle: true,          // Shuffle data on every retries
                batchSize: batchSize,   // for stochastic gradient descent, calculate gradient for the whole dataset
            });
            const loss = history.history.loss[0].toFixed(6);
            const acc = history.history.acc[0].toFixed(4);
            console.log(`Step: ${step++}: loss: ${loss}, accuracy: ${acc}`);
        }
    }
    totalTimer.end();
    console.log(`Model training completed. Cost: ${totalTimer.seconds().toFixed(2)}s`);

    // // Calculate accuracy on test set
    let xt = tf.tensor(testImages).reshape([5000, 28, 28, 1]);
    let yt = tf.oneHot(tf.tensor1d(testLabels, 'int32'), 10).toFloat();
    
    let ytp = model.predict(xt) as tf.Tensor;
    let predictions = ytp.argMax(1).dataSync();
	let labels = yt.argMax(1).dataSync();

	let correct = 0;
	for (let i = 0; i < labels.length; i++) {
		if (predictions[i] === labels[i]) {
			correct++;
		}
	}
	const accuracy = ((correct / labels.length) * 100).toFixed(2);
	console.log(`Test set accuracy: ${accuracy}%\n`);
}

async function loadImages(path, numRecords) {
    let config = {
        path, bytesOffset: 16, bytesStep: 1, shouldScale: true, numRecords
    }
    return await DataHelper.readNumericsFromFile(config);
}

async function loadLabels(path, numRecords) {
    let config = {
        path, bytesOffset: 8, bytesStep: 1, shouldScale: false, numRecords
    }
    return await DataHelper.readNumericsFromFile(config);
}

function initModel(): tf.Sequential {
    let model = tf.sequential();
    let hidden1 = tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    });
    model.add(hidden1);

    let pool1 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    });
    model.add(pool1);

    let hidden2 = tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    });
    model.add(hidden2);

    let pool2 = tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    });
    model.add(pool2);

    model.add(tf.layers.flatten());
    let dense = tf.layers.dense({
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    });
    model.add(dense);

    let LearningRate = 0.15;
    model.compile({
        optimizer: tf.train.sgd(LearningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

main();