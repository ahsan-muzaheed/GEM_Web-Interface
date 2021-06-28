const multer = require('multer');
const path= require('path');
const fileStorageEngine = multer.diskStorage({
    
    destination: (req, file, cb) => {
        cb(null, __dirname + '/../uploads');
    },
    filename: (req, file, cb) => {
        
        //cb(null, Date.now() + '--' + file.originalname);
        cb(null, file.fieldname + path.parse(file.originalname).ext);
    } 
});
const upload = multer( {storage : fileStorageEngine});
//console.log('Multer reach');
module.exports = upload;