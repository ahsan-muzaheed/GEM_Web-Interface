const { response } = require("express");
const fs = require("fs");
const path = require("path");
const fileInput = require("./fileRead");
// Load basic GET pages
exports.getLandingPage = (req, res, next) => {
  console.log("Landing called!");

  res.render("landing", {
    pageTitle: "Landing",
  });
};

exports.getUploadResultPage = (req, res, next) => {
  console.log("uploadResult called!");
  let returnedData = "";

  // Read matrix in txt file
  const targetPath = path.join(
    __dirname,
    "..",
    "python",
    "results",
    "num_of_nodes.txt"
  );
  returnedData = fs.readFileSync(targetPath, "utf8");

  console.log(returnedData);
  resultData =
    '<div id="resultcontent"><pre>' +
    returnedData.replace("/\n/g", "<br/>") +
    "</pre>";
  res.render("uploadResult", {
    pageTitle: "uploadResult",
    pageData: resultData,
  });
};

exports.getLoadingPage = (req, res, next) => {
  console.log("loading page called!");
  res.render("loading", {
    pageTitle: "Loading",
  });
};

exports.getGraphEmbeddingPage = (req, res, next) => {
  res.render("graphEmbedding", {
    pageTitle: "GraphEmbedding",
  });
};

exports.embeddingHandler = (req, res, next) => {
  // Call embedding algorithm here
  console.log(req.body.embeddingMethod);
  console.log(req.body.dimension);

  // Laplacian
  if (req.body.embeddingMethod == "Laplacian") {
    const { spawn } = require("child_process");
    const dimension = req.body.dimension;
    const ls = spawn("python3", ["python/Laplacian.py", dimension]);

    ls.stdout.on("data", (data) => {
      console.log(`stdout: ${data}`);
    });

    ls.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });

    ls.on("close", (code) => {
      console.log("Laplacian called");
      //ls.stdin.pause();
      //ls.kill();

      console.log(`graph embedding child process exited with code ${code}`);
      //process.exit();
      res.redirect("/downStreamML");
    });
  }

  //res.redirect("/downStreamML");
};

exports.getDownStreamMLPage = (req, res, next) => {
  res.render("downStream", {
    pageTitle: "downStream",
  });
};

exports.downstreamHandler = (req, res, next) => {
  const obj = JSON.parse(JSON.stringify(req.body));
  console.log("Machine Learning Option: ", obj);
  let data = JSON.stringify(obj);
  // fs.writeFileSync(
  //   path.join(__dirname, "..", "python", "results", "downStreamMLinfo.json"),
  //   data
  // );
  // Call python here
  const { spawn } = require("child_process");
  const cp = spawn("python3", ["./python/downStreamML.py", data]);
  cp.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });
  cp.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });
  cp.on("close", (code) => {
    console.log("downstream finished");
    console.log(`downstream child process exited with code ${code}`);
    res.redirect("/result");
  });
};

exports.getResultPage = (req, res, next) => {
  console.log("result page called");

  let resultData = fileInput.readResultTxt("result", ".txt");
  let resultVal = "";
  resultData.then((result) => {
    let htmlResult =
      '<div id="resultcontent"><pre>' +
      result.toString("utf8").replace(/\n/g, "<br />") +
      "</pre>";
    console.log(result.toString("utf8"));
    resultVal = htmlResult;
    res.render("result", {
      pageTitle: "result",
      result: resultVal,
    });
  });
  // res.render("result", {
  //   pageTitle: "result",
  //   result: resultVal,
  // });
};
