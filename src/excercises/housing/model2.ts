#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import DataHelper from '../../helpers/DataHelper';

import '@tensorflow/tfjs-node';
// import '@tensorflow/tfjs-node-gpu';

main();
async function main() {
    let features_data: any[] = [];
    let labels_data: any[] = [];

    let dataPath = 'database/housing.csv';
    let csvData = await DataHelper.readCsv(dataPath);

    csvData.forEach(rowData => {
        if (!rowData || !rowData.length || !rowData[0])
            return;
        else {
            labels_data.push(Math.floor(rowData[rowData.length - 1] / 1000));
            features_data.push(rowData[5] / 1000); // Remove frist 2 features lat,long
        }
    });

    // Define the linear model
    function predict(x) {
        return tf.tidy(() => {
            return a.mul(x.square()).add(b.mul(x)).add(c);
        });
    }

    // Define a loss function
    function loss(predictions, labels) {
        const error = predictions.sub(labels).square().mean();
        console.log('Current loss:', Math.sqrt(error.dataSync()[0]));
        return error;
    }

    // Define training loop with an optmizer
    async function train(xs, ys, numIterations = 75) {
        for (let iter = 0; iter < numIterations; iter++) {
            // console.log('Interation ', iter);
            optimizer.minimize(() => {
                const pred = predict(xs);
                return loss(pred, ys);
            });
        }
    }

    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));

    // // Setup optimizer
    // const learningRate = 0.0002;
    // const optimizer = tf.train.sgd(learningRate);

    // let xs = tf.tensor1d([1,2,3,4,5,6,7,8,9,10]);
    // let ys = tf.tensor1d([3,5,7,9,11,13,15,17,19,21]);

    // await train(xs, ys, 5000);

    // console.log('Model training completed');

    // console.log('Prediction for [1,3,5,7]: ');
    // (predict(tf.tensor1d(
    //     [1, 3, 5, 7],
    // ))).print();

    // console.log('Accurate Result is: [3, 7, 11, 15]');

    // Setup optimizer
    const learningRate = 0.004;
    const optimizer = tf.train.sgd(learningRate);

    let xs = tf.tensor1d(features_data);
    let ys = tf.tensor1d(labels_data);

    await train(xs, ys, 1000);

    console.log('Model training completed');

    console.log('Accurate ys is');
    ys.print();

    console.log('Prediction for xs');
    (predict(xs)).print();
}


