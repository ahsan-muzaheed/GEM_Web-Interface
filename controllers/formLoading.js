const { response } = require("express");
const fs = require("fs");
const path = require("path");

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

  res.render("uploadResult", {
    pageTitle: "uploadResult",
    pageData: returnedData,
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
      //console.log(`child process exited with code ${code}`);

      //res.redirect('/uploadResult');
    });
  }

  res.redirect("/downStreamML");
};

exports.getDownStreamMLPage = (req, res, next) => {
  res.render("downStream", {
    pageTitle: "downStream",
  });
};
exports.getResultPage = (req, res, next) => {
  res.render("result", {
    pageTitle: "result",
  });
};
