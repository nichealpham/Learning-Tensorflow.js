import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node-gpu';

export class ModelConfig {
    epochs?: number;
    shuffle?: boolean;
    optimizer?: string;
    learningRate?: number;
}

export class Model {
    public maxPower: number;
    public numSamples: number;
    public numFeatures: number;
    public pShape: tf.Tensor;
    public coeffs: tf.Variable;
    public config: ModelConfig;

    constructor(config?: ModelConfig) {
        if (config)
            this.config = {...{epochs: 5, shuffle: true, optimizer: 'sgd', learningRate: 0.005}, ...config};
    }

    async train(xs: number[][], ys: number[], pShape: number[][], epochsSuccessCallback?: any): Promise<void> {
        this.maxPower = pShape.length;
        this.numSamples = ys.length;
        this.numFeatures = xs[0].length;
        // Prepare power shape and data
        let Y = tf.tensor(ys).reshape([this.numSamples, 1]);
        let X = tf.tensor(xs).reshape([this.numSamples, this.numFeatures]);
        this.pShape = tf.tensor(pShape).reshape([this.maxPower, this.numFeatures]);
        // Prepare weights and bias
        this.coeffs = tf.variable(tf.zeros([this.maxPower, this.numFeatures]));

        let optimizer = tf.train[this.config.optimizer!](this.config.learningRate);
        for (let i = 0; i < this.config.epochs!; i++) {
            optimizer.minimize(() => {
                let loss = cost(predict(X, this.pShape, this.coeffs, this.numFeatures, this.maxPower), Y);
                if (epochsSuccessCallback)
                    epochsSuccessCallback(i, loss.dataSync()[0]);
                return loss;
            });
        };
    }

    predict(xs: number[][]) {
        return predict(tf.tensor(xs).reshape([xs.length, xs[0].length]), this.pShape, this.coeffs, xs[0].length, this.maxPower);
    }

    reConfig(config) {
        this.config = {...this.config, ...config};
    }
}

function predict(xs: tf.Tensor, ps: tf.Tensor, cof: tf.Variable, nfe: number, max: number): tf.Tensor {
    return tf.tidy(() => {
        let rs: any = null;
        for (let i = 0; i < max; i++) {
            let msk: number[] = [];
            for (let j = 0; j < max; j++) {
                j === i ? msk.push(1) : msk.push(0);
            }
            let pts = xs.pow(tf.scalar(max - i - 1)).dot((tf.tensor(msk, [1, max]).dot(cof.mul(ps))).transpose());
            rs ? rs.add(pts) : rs = pts;
        }
        return rs;
    });
}

function cost(predicts: tf.Tensor, labels: tf.Tensor): tf.Tensor {
    return predicts.sub(labels).square().mean();
}