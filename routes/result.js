const path = require('path');
const express = require('express');
const router = express.Router();
const formController = require('../controllers/formLoading');

router.get('/result', formController.getResultPage);

module.exports = router;