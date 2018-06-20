#!/usr/bin/env node
import * as tf from "@tensorflow/tfjs";

main();
async function main() {
    let model = tf.sequential();

    let hidden1 = tf.layers.dense({
        units: 4,
        inputShape: [3],
        activation: 'sigmoid'
    });
    model.add(hidden1);
    
    let hidden2 = tf.layers.dense({
        units: 2,
        activation: 'sigmoid'
    });
    model.add(hidden2);
    
    let output = tf.layers.dense({
        units: 3,
        activation: 'softmax'
    });
    model.add(output);
    
    model.compile({
        optimizer: tf.train.sgd(2.1),
        loss: tf.losses.meanSquaredError
    });

    let xs = tf.tensor2d([
        [1, 1, 1],
        [0.5 , 0.5, 0.5],
        [0, 0, 0]
    ]);
    let ys = tf.tensor2d([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]);

    for (let i = 0; i < 250; i++) {
        let trainingProcess = await model.fit(xs, ys, {
            epochs: 10,
            shuffle: true,
            batchSize: 64,
        });
        console.log("Loss after Epoch " + i + " : " + trainingProcess.history.loss[0]);
    }
    console.log('Model training completed');

    ((model.predict(xs)) as tf.Tensor).print();

    ((model.predict(tf.tensor2d([
        [1, 1, 1],
    ]))) as tf.Tensor).print();
    
    ((model.predict(tf.tensor2d([
        [0.5 , 0.5, 0.5],
    ]))) as tf.Tensor).print();

    ((model.predict(tf.tensor2d([
        [0, 0, 0]
    ]))) as tf.Tensor).print();
    
}
