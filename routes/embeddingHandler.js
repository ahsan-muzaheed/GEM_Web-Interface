const express = require("express");
const router = express.Router();
const formController = require("../controllers/formLoading");

router.post("/embeddingHandler", formController.embeddingHandler);

module.exports = router;
