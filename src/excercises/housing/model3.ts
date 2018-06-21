#!/usr/bin/env node
import FileHelper from '../../helpers/FileHelper';
import * as tf from '@tensorflow/tfjs';
import * as Regressor from '../../models/LinearRegressor'; //eslint-disable-line

import '@tensorflow/tfjs-node';
// import '@tensorflow/tfjs-node-gpu';

main();
async function main() {
    let csvData = await FileHelper.readCsv('db/housing.csv');
    let labels_data: any[] = [];
    let features_data: any[] = [];

    csvData.forEach(rowData => {
        if (!rowData || !rowData.length || !rowData[0])
            return;
        else {
            labels_data.push(Math.floor(rowData[rowData.length - 1] / 1000));
            features_data.push([rowData[2] / 10, rowData[3] / 1000, rowData[4] / 1000,  rowData[5] / 1000, rowData[6] / 200]); // Remove frist 2 features lat,long
        }
    });

    let model = new Regressor.Model({
        epochs: 4000,
        shuffle: true,
        learningRate: 0.004,
    });

    await model.train(features_data, labels_data, (i, cost) => {
        console.log(`Epoch ${i} loss is: ${cost}`,);
    });

    // await model.save('housing/model3');
    console.log('Model training completed');

    tf.tensor2d(features_data).print();
    console.log('Accurate ys is');
    tf.tensor2d(labels_data, [labels_data.length, 1]).print();

    console.log('Prediction for xs');
    (model.predict(features_data)).print();
}


