import * as fs from 'fs';
import * as parse from 'csv-parse';

class FileHelper {
    static async readCsv(path: string): Promise<any[]> {
        let csvData: any[] = [];
        return new Promise<any>((resolve, reject) => {
            fs.createReadStream(path).pipe(parse({delimiter: ','}))
                .on('data', (csvrow: any) => {
                    csvrow = csvrow.map(value => Number(value));
                    csvData.push(csvrow);        
                })
                .on('end', () => {
                    resolve(csvData);
                })
                .on('error', (err) => {
                    reject(err);
                });
        });
    }

    static async readBuffer(path: string): Promise<any> {
        return new Promise((resolve, reject) => {
            fs.readFile(path, (err, data) => {
                err ? reject(err) : resolve(data);
            });
        });
    }

    // One buffer contains a set of images, read each image into an array of pixels array
    static async readImagesFromBuffer(
            path: string, 
            offsetBytes: number = 0, 
            imgDimension: {width: number, height: number} = {width: 1, height: 1},
            pixelBytes: number = 1,
            downScale: boolean = true): Promise<any[]> {

        let images: any[] = [];
        let buffer = await this.readBuffer(path);

        // imgPixels: flat array of pixel depths
        let imgPixels = imgDimension.width * imgDimension.height;
        let downScaler = downScale ? 1 / (Math.pow(2, 8 * pixelBytes) - 1) : 1;
        let index = offsetBytes;
        while (index < buffer.byteLength) {
            let arr = new Float32Array(imgPixels);
            for (let i = 0; i < imgPixels; i++) {
                // For 1 byte => readUnit8. For >= 2 bytes => readUInt8*iBE 
                arr[i] = buffer[`readUInt${8 * pixelBytes}${pixelBytes >= 2 ? 'BE' : ''}`](index) * downScaler;
                index += pixelBytes;
            };
            images.push(arr);
        }
        return images;
    }

    // One buffer contains a set of labels collumns => read them into array of labels array
    static async readLabelsFromBuffer(
            path: string,
            offsetBytes: number = 0,
            labelColumns: number = 1,
            labelBytes: number = 1,
            downScale: boolean = false): Promise<any[]> {
        
        // Infact, we can construct this function using readImagesFromBuffer, very similar
        let labelDimension = {
            width: labelColumns,
            height: 1
        };   // construct a 1 * n matrics represent a row of labels similar to an image metric
        return await this.readImagesFromBuffer(path, offsetBytes, labelDimension, labelBytes, downScale);
    }
};

Object.seal(FileHelper);
export default FileHelper;