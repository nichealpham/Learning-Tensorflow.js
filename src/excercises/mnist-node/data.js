/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');
import '@tensorflow/tfjs-node-gpu';

const assert = require('assert');
const fs = require('fs');
const https = require('https');
const util = require('util');
const zlib = require('zlib');
const FileHelper = require('../../helpers/FileHelper');

const readFile = util.promisify(fs.readFile);

// MNIST data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'db/mnist/train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'db/mnist/train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = 'db/mnist/t10k-images-idx3-ubyte';
const TEST_LABELS_FILE = 'db/mnist/t10k-labels-idx1-ubyte';
const IMAGE_HEADER_BYTES = 16;
const IMAGE_DIMENSION_SIZE = 28;
const IMAGE_FLAT_SIZE = IMAGE_DIMENSION_SIZE * IMAGE_DIMENSION_SIZE;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

// Downloads a test file only once and returns the buffer for the file.
async function fetchOnceAndSaveToDiskWithBuffer(filename) {
	return new Promise(resolve => {
		const url = `${BASE_URL}${filename}.gz`;
		if (fs.existsSync(filename)) {
			resolve(readFile(filename));
			return;
		}
		const file = fs.createWriteStream(filename);
		console.log(`  * Downloading from: ${url}`);
		https.get(url, (response) => {
			const unzip = zlib.createGunzip();
			response.pipe(unzip).pipe(file);
			unzip.on('end', () => {
				resolve(readFile(filename));
			});
		});
	});
}

// Shuffles data and label using Fisher-Yates algorithm.
function shuffle(data, label) {
	let counter = data.length;
	let temp = 0;
	let index = 0;
	while (counter > 0) {
		index = (Math.random() * counter) | 0;
		counter--;
		// data:
		temp = data[counter];
		data[counter] = data[index];
		data[index] = temp;
		// label:
		temp = label[counter];
		label[counter] = label[index];
		label[index] = temp;
	}
}

async function loadImages(filename) {
	let offsetBytes = 16; 	// Dataset contain 4 collums, each collum is 4 bytes => offset 16 bytes
	let pixelBytes = 1;		// Each pixel is 1 bytes in depth
	let imgDimension = {
		width: 28,
		height: 28
	}; 						// image is 28 * 28 pixel
	let downScale = true;	// Should normallize each value into range 0 -> 1 float32

	return await FileHelper.default.readImagesFromBuffer(filename, offsetBytes, imgDimension, pixelBytes, downScale);
}

async function loadLabels(filename) {
	let offsetBytes = 8;	// Labels data contain 2 collumn, each collum 4 bytes => offset 8 bytes
	let labelColumns = 1;	// We only have 1 label
	let labelBytes = 1;		// Each label is 1 byte in depth
	let downScale = false;	// Label range from int 1 -> int 10 => so dont scale it into 0 -> 1

	return await FileHelper.default.readLabelsFromBuffer(filename, offsetBytes, labelColumns, labelBytes, downScale);
}

/** Helper class to handle loading training and test data. */
class MnistDataset {
	constructor() {
		this.dataset = null;
		this.trainSize = 0;
		this.testSize = 0;
		this.trainBatchIndex = 0;
		this.testBatchIndex = 0;
	}

	/** Loads training and test data. */
	async loadData() {
		this.dataset = await Promise.all([
			loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE),
			loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
		]);
		this.trainSize = this.dataset[0].length;
		this.testSize = this.dataset[2].length;

		// Shuffle training and test data:
		shuffle(this.dataset[0], this.dataset[1]);
		shuffle(this.dataset[2], this.dataset[3]);
	}

	/** Resets training data batches. */
	resetTraining() {
		this.trainBatchIndex = 0;
	}

	/** Resets test data batches. */
	resetTest() {
		this.testBatchIndex = 0;
	}

	/** Returns true if the training data has another batch. */
	hasMoreTrainingData() {
		return this.trainBatchIndex < this.trainSize;
	}

	/** Returns true if the test data has another batch. */
	hasMoreTestData() {
		return this.testBatchIndex < this.testSize;
	}

	/**
	 * Returns an object with training images and labels for a given batch size.
	 */
	nextTrainBatch(batchSize) {
		return this._generateBatch(true, batchSize);
	}

	/**
	 * Returns an object with test images and labels for a given batch size.
	 */
	nextTestBatch(batchSize) {
		return this._generateBatch(false, batchSize);
	}

	_generateBatch(isTrainingData, batchSize) {
		let batchIndexMax;
		let size;
		let imagesIndex;
		let labelsIndex;
		if (isTrainingData) {
			batchIndexMax = this.trainBatchIndex + batchSize > this.trainSize ?
				this.trainSize - this.trainBatchIndex :
				batchSize + this.trainBatchIndex;
			size = batchIndexMax - this.trainBatchIndex;
			imagesIndex = 0;
			labelsIndex = 1;
		} else {
			batchIndexMax = this.testBatchIndex + batchSize > this.testSize ?
				this.testSize - this.testBatchIndex :
				batchSize + this.testBatchIndex;
			size = batchIndexMax - this.testBatchIndex;
			imagesIndex = 2;
			labelsIndex = 3;
		}

		// Only create one big array to hold batch of images.
		const imagesShape = [size, IMAGE_DIMENSION_SIZE, IMAGE_DIMENSION_SIZE, 1];
		const images = new Float32Array(tf.util.sizeFromShape(imagesShape));

		const labelsShape = [size, 1];
		const labels = new Int32Array(tf.util.sizeFromShape(labelsShape));

		let imageOffset = 0;
		let labelOffset = 0;

		while ((isTrainingData ? this.trainBatchIndex : this.testBatchIndex) <
			batchIndexMax) {

			images.set(this.dataset[imagesIndex][this.trainBatchIndex], imageOffset);
			labels.set(this.dataset[labelsIndex][this.trainBatchIndex], labelOffset);

			isTrainingData ? this.trainBatchIndex++ : this.testBatchIndex++;
			imageOffset += IMAGE_FLAT_SIZE;
			labelOffset += 1;
		}

		return {
			image: tf.tensor4d(images, imagesShape),
			label: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
		};
	}
}

module.exports = new MnistDataset();
