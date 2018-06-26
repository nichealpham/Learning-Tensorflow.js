import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node-gpu';
import DataHelper from '../../helpers/DataHelper';
// import '@tensorflow/tfjs-node-gpu';

main();
async function main() {
    let features_data: any[] = [];
    let labels_data: any[] = [];

    let dataPath = 'db/housing.csv';
    let csvData = await DataHelper.readNumericsFromCsv(dataPath);

    csvData.forEach(rowData => {
        if (!rowData || !rowData.length || !rowData[0])
            return;
        else {
            let label = rowData[rowData.length - 1];
            if (label < 30000)
                labels_data.push([1, 0, 0]);
            else if (label < 90000)
                labels_data.push([0, 1, 0]);
            else
                labels_data.push([0, 0, 1]);

            rowData.splice(-1); // Remove label collums
            features_data.push(rowData.splice(2, 10)); // Remove frist 2 features lat,long
        }
    });

    let model = await initModel();
    
    tf.tensor2d(features_data).print();
    tf.tensor2d(labels_data).print();
    
    let xs = tf.tensor2d(features_data);
    let ys = tf.tensor2d(labels_data);

    for (let i = 0; i < 4; i++) {
        let trainingProcess = await model.fit(xs, ys, {
            epochs: 5,
            shuffle: true,
            batchSize: 64,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    console.log(`Epoch ${epoch}: loss = ${log!.loss}`);
                }
            }
        });
        console.log("Loss after Epoch " + i + " : " + trainingProcess.history.loss[0]);
    }
    console.log('Model training completed');

    console.log('Predict for some xs: ');
    tf.tensor2d(features_data).print()

    console.log('Expected value for ys:');
    tf.tensor2d(labels_data).print()

    console.log('Predicted value for ys');
    (model.predict(tf.tensor2d([
        [15, 5612 , 1283, 1015 , 472 , 1.4936],
        [19, 7650 , 1901, 1129 , 463 , 1.82],
        [17, 720  , 174 , 333  , 117 , 1.6509],
        [17, 2677 , 531 , 1244 , 456 , 3.0313001],
        [19, 2672 , 552 , 1298 , 478 , 1.9797],
        [52, 1820 , 300 , 806  , 270 , 3.0146999]
    ])) as tf.Tensor).print()
}

async function initModel() {
    let model = tf.sequential();

    let hidden1 = tf.layers.dense({
        units: 6,
        inputShape: [6],
        activation: 'sigmoid'
    });
    model.add(hidden1);
    
    let hidden2 = tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
    });
    model.add(hidden2);

    let hidden3 = tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
    });
    model.add(hidden3);

    let output = tf.layers.dense({
        units: 3,
        activation: 'softmax'
    });
    model.add(output);
    
    model.compile({
        optimizer: tf.train.sgd(0.005),
        loss: tf.losses.meanSquaredError
    });
    return model;
}
