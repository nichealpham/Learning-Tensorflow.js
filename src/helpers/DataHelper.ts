import * as fs from 'fs';
import * as parse from 'csv-parse';

class DataHelper {
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
};

Object.seal(DataHelper);
export default DataHelper;