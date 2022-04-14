// GEM Web-Application
// Written by Jun Yong Lee
// NMSU

const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const app = express();

const ladningRoutes = require("./routes/landing");
const uploadResult = require("./routes/uploadResult");
const loading = require("./routes/loading");
const graphEmbedding = require("./routes/graphEmbedding");
const downStreamML = require("./routes/downStreamML");
const result = require("./routes/result");
const embeddingHandler = require("./routes/embeddingHandler");

const PORT = process.env.PORT || 8080;
app.set("view engine", "ejs");
app.set("views", "views");

// Body Parser
app.use(bodyParser.urlencoded({ extended: false }));
// Static assets
app.use(express.static(path.join(__dirname, "public")));
app.use(express.static(path.join(__dirname, "python", "results", "rocCurve")));

// Page Routings
app.use(ladningRoutes);
app.use(uploadResult);
//app.use(loading);
app.use(graphEmbedding);
app.use(embeddingHandler);
app.use(downStreamML);
app.use(result);

app.listen(PORT, () => {
  console.log(`server started on ${PORT}`);
});
