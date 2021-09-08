const path = require("path");
const express = require("express");
const router = express.Router();
const formController = require("../controllers/formLoading");
//const down = require("../controllers/downstreamFrom");

router.get("/downStreamML", formController.getDownStreamMLPage);
router.post("/downstreamHandler", formController.downstreamHandler);
//router.post("/downstreamHandler", down.downstreamHandler);

module.exports = router;
