#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import * as Regressor from '../../models/PolynomialRegressor'; //eslint-disable-line

import '@tensorflow/tfjs-node-gpu';
// import '@tensorflow/tfjs-node-gpu';

main();
async function main() {
    let features_data: number[][] = [
        [1, 2, 3], 
        [3, 1, 2], 
        [2, 1, 2], 
        [3, 4, 1]
    ]; // -> X
    let labels_data: number[] = [
        48,
        44,
        33,
        22
    ]; // -> Y
    let powerShape = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]; // -> Y = AX^2 + BX + C

    let model = new Regressor.Model({
        epochs: 2000,
        shuffle: true,
        learningRate: 0.0001,
    });

    await model.train(features_data, labels_data, powerShape, (i, cost) => {
        console.log(`Epoch ${i} loss is: ${cost}`);
    });

    // await model.save('housing/model3');
    console.log('Model training completed');

    tf.tensor2d(features_data).print();
    console.log('Accurate ys is');
    tf.tensor2d(labels_data, [labels_data.length, 1]).print();

    console.log('Prediction for xs');
    (model.predict(features_data)).print();
}


