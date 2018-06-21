import * as fs from 'fs';
import * as util from 'util';
import * as parse from 'csv-parse';

const readFile = util.promisify(fs.readFile);

class FileHelper {
    static async readCsv(path): Promise<any> {
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

    static async readAsBuffer(path): Promise<any> {
        return new Promise((resolve, reject) => {
            fs.readFile(path, (err, data) => {
                if (err) reject(err);
                else resolve(data);
            });
        });
    }

    // static async loadImages(path) {
    //     let buffer = await this.readAsBuffer(path);
    //     let downsize = 1.0 / 255.0;
    
    //     let images = [];
    //     let index = headerBytes;
    //     while (index < buffer.byteLength) {
    //         const array = new Float32Array(recordBytes);
    //         for (let i = 0; i < recordBytes; i++) {
    //             array[i] = buffer.readUInt8(index++) * downsize;
    //         }
    //         images.push(array);
    //     }
    
    //     assert.equal(images.length, headerValues[1]);
    //     return images;
    // }
};

Object.seal(FileHelper);
export default FileHelper;