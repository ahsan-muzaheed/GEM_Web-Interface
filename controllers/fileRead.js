const fs = require('fs');
const util = require('util');
const readFile = util.promisify(fs.readFile);
const path = require('path');


exports.run = async (filename, ext) => {
    
    const data = await readFile(path.join(__dirname, '..', 'uploads', filename + ext));
    //console.log(data.toString());
    console.log('reached readfile');

}





