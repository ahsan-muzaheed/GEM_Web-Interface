const express = require("express");
const router = express.Router();

const formController = require("../controllers/formLoading");

const upload = require("../util/multer");
const fileRead = require("../controllers/fileRead");

router.get("/uploadResult", formController.getUploadResultPage);
router.post("/postUploadResultPage", formController.postUploadResultPage);
module.exports = router;
