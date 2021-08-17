const express = require('express');
const router = express.Router();

const formController = require('../controllers/formLoading');

const upload = require('../util/multer');
const fileRead = require('../controllers/fileRead');


//const edgeListHandler = require('../controllers/edgeListHandler');



router.get('/', formController.getLandingPage);

//router.get('/loading', formController.getLoadingPage);
//const uploadFields = upload.fields([{name:'edgeList'}, {name:'nodeFeature'},{name:'labels'}]);

router.post('/single', upload.fields([{name:'adjList'}, {name:'nodeFeature'}, {name:'labels'}]), (req, res) => {


    
    // Call python and start processing the read files
    if(req.files.adjList || req.files.nodeFeature || req.files.labels){
        //fileRead.run('edgeList', '.csv');
        //edgeListHandler.run();
        const { spawn } = require('child_process');
        const ls = spawn('python3', ['./python/functionDefs.py']);
        ls.stdout.on('data', (data) => {
            
            console.log(`stdout: ${data}`);
        
             
        });
    
        ls.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });
    
        ls.on('close', (code) => {
            const fs = require('fs');
            const path = require('path');
            const dir = __dirname;
            const targetPath = path.join(dir, '..', '/uploads');
            const length = fs.readdirSync(targetPath).length;
            console.log(`child process exited with code ${code}`);
            if(length >= 1){
                console.log(length);
            }
            res.redirect('/uploadResult');
        });

        
    }

    //res.redirect('/loading');

    
    
    //fileRead.run('edgeList', '.py');
    //console.log(data);
    //res.send('Single file upload success');
    
});



module.exports = router;