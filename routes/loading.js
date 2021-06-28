const path = require('path');
const express = require('express');
const router = express.Router();
const formController = require('../controllers/formLoading');
const fs = require('fs');
const dir = __dirname;
const targetPath = path.join(dir, '..', '/uploads');
const length = fs.readdirSync(targetPath).length;


router.get('/loading', formController.getLoadingPage);




module.exports = router;