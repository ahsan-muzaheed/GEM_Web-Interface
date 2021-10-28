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
  let isMultiClass = false;
  // Read matrix in txt file
  const targetPath = path.join(
    __dirname,
    "..",
    "python",
    "results",
    "num_of_nodes.txt"
  );
  returnedData = fs.readFileSync(targetPath, "utf8");
  numOfClass = returnedData.split(" ")[1];

  console.log(returnedData);
  resultData =
    '<div id="resultcontent"><pre>' +
    returnedData.replace("/\n/g", "<br/>") +
    "</pre>";
  res.render("uploadResult", {
    pageTitle: "uploadResult",
    pageData: resultData,
    numOfClass: numOfClass,
  });
};

exports.postUploadResultPage = (req, res, next) => {
  console.log(req.body);
  let posVal = req.body["posVal"];

  if (req.body["binarize"] === "yes") {
    // binarize label.npy
    const { spawnSync } = require("child_process");
    const ls = spawnSync("python3", ["python/binarize.py", posVal]);

    // ls.stdout.on("data", (data) => {
    //   console.log(`stdout: ${data}`);
    // });

    // ls.stderr.on("data", (data) => {
    //   console.error(`stderr: ${data}`);
    // });

    // ls.on("close", (code) => {
    //   //res.redirect("graphEmbedding");
    //   //res.render("graphEmbedding", { result: htmlResult });
    // });
  } else {
    res.redirect("graphEmbedding");
  }

  let resultData = fileInput.readResultTxt("binarizedResult", ".txt");

  let resultVal = "";
  resultData.then((result) => {
    let htmlResult =
      '<div id="binarizedContent"><pre>' +
      result.toString("utf8").replace(/\n/g, "<br />") +
      "</pre>";
    //console.log(result.toString("utf8"));
    resultVal = htmlResult;
    console.log("resultVal: ", resultVal);

    res.render("graphEmbedding", {
      pageTitle: "GraphEmbedding",
      resultVal: resultVal,
    });
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
  console.log(req.body);
  // Laplacian
  if (req.body.embeddingMethod == "Laplacian") {
    console.log(req.body.embeddingMethod);
    console.log(req.body.dimension);
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
  } else if (req.body.embeddingMethod == "GAT") {
    let param = JSON.stringify(req.body);
    //const { spawn } = require("child_process");
    //const ls = spawn("python3", ["python/GAT.py", param]);
    const { spawnSync } = require("child_process");
    const ls = spawnSync("python3", ["python/GAT.py", param]);
    console.log("stdout here: \n" + ls.stdout);
    // ls.stdout.on("data", (data) => {
    //   console.log(`stdout: ${data}`);
    // });

    // ls.stderr.on("data", (data) => {
    //   console.error(`stderr: ${data}`);
    // });

    // ls.on("close", (code) => {
    //   console.log("GAT called");

    //   console.log(`graph embedding child process exited with code ${code}`);
    //   res.redirect("/downStreamML");
    // });
    res.redirect("/result");
  } else {
    res.redirect("/downStreamML");
  }
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
    //console.error(`stderr: ${data}`);
    //res.redirect("/result");
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

    let method = result.toString("utf8").split("\n")[3].split(" ")[1];
    let model = result.toString("utf8").split("\n")[5].split(/\s+/)[2];
    if (model == "LogisticRegression()") model = "LogisticRegression";
    if (model == "SVC(probability=True)") model = "SVC";
    if (model == "KNeighborsClassifier()") model = "KNeighborsClassifier";
    if (model == "DecisionTreeClassifier()") model = "DecisionTreeClassifier";
    console.log("model: ", model);
    console.log("method: ", method);

    resultVal = htmlResult;

    let imagefile = [];
    fs.readdirSync(
      path.join(__dirname, "..", "python/results/rocCurve")
    ).forEach((file) => {
      if (file.slice(-3) == "png") {
        imagefile.push(file);
      }
    });
    console.log(imagefile);
    res.render("result", {
      pageTitle: "result",
      result: resultVal,
      graph: imagefile,
      method: method,
      model: model,
    });
  });
};
