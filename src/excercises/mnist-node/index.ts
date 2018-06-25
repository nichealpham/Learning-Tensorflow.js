import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
tf.setBackend('tensorflow');

import * as fs from 'fs';
import DataHelper from '../../helpers/DataHelper';

async function main() {
    let model: tf.Sequential = initModel();
    // Download if have no data
    if (!fs.existsSync('./db/mnist-db'))
        await DataHelper.downloadFilesFromUrls([
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
		    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ], './db/mnist-db');
    // // Then load each of them into an array of numerics
    let trainImages = await loadImages('./db/mnist-db/train-images-idx3-ubyte', 20000 * 784);
    let trainLabels = await loadLabels('./db/mnist-db/train-labels-idx1-ubyte', 20000 * 1);
    let testImages = await loadImages('./db/mnist-db/t10k-images-idx3-ubyte', 5000 * 784);
    let testLabels = await loadLabels('./db/mnist-db/t10k-labels-idx1-ubyte', 5000 * 1);
    // // Prepare data before training
    let xs = tf.tensor(trainImages).reshape([20000, 28, 28, 1]);
    let ys = tf.oneHot(tf.tensor1d(trainLabels, 'int32'), 10).reshape([20000, 10]);
    let xt = tf.tensor(testImages).reshape([5000, 28, 28, 1]);
    let yt = tf.tensor(testLabels).reshape([5000, 1]);
    // Now we train model with the input xs, ys => test accuracy on xt, yt
    await model.fit(xs, ys, {
        epochs: 10,
        shuffle: true,
        batchSize: 32,
        callbacks: {
            onEpochEnd: async (epoch, log) => {
                console.log(`Epoch ${epoch}: loss = ${log!.loss}`);
            }
        }
    });
    console.log('Model training completed');

    // // Calculate accuracy on test set
    // let ytp = model.predict(xt) as tf.Tensor;
    // let loss = yt.sub(ytp).square().mean();

    // console.log('Mean squared error on testing set is', loss.print());

    // let predictions = ytp.argMax(1).dataSync();
	// let labels = yt.argMax(1).dataSync();

	// let correct = 0;
	// for (let i = 0; i < labels.length; i++) {
	// 	if (predictions[i] === labels[i]) {
	// 		correct++;
	// 	}
	// }
	// const accuracy = ((correct / labels.length) * 100).toFixed(2);
	// console.log(`Test set accuracy: ${accuracy}%\n`);
}

async function loadImages(path, numRecords) {
    let config = {
        path, bytesOffset: 4, bytesStep: 1, shouldScale: true, numRecords
    }
    return await DataHelper.readNumericsFromFile(config);
}

async function loadLabels(path, numRecords) {
    let config = {
        path, bytesOffset: 2, bytesStep: 1, shouldScale: false, numRecords
    }
    return await DataHelper.readNumericsFromFile(config);
}

function initModel(): tf.Sequential {
    let model = tf.sequential();
    let hidden1 = tf.layers.conv2d({
        filters: 8,
        kernelSize: [5, 5],
        inputShape: [28, 28, 1],
        strides: [1, 1],
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
        filters: 16,
        kernelSize: [5, 5],
        strides: [1, 1],
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

    let LearningRate = 1.5;
    model.compile({
        optimizer: tf.train.sgd(LearningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

main();