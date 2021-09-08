const fs = require("fs");
const path = require("path");

exports.downstreamHandler = (req, res, next) => {
  console.log("Calling here!");
  const obj = JSON.parse(JSON.stringify(req.body));
  console.log("Machine Learning Option: ", obj);
  const data = JSON.stringify(obj);
  fs.writeFileSync(
    path.join(__dirname, "..", "python", "results", "downStreamMLinfo.json"),
    data
  );

  // Call python here
  const { exec } = require("child_process");

  exec("python3 python/downStreamML.py", (error, stdout, stderr) => {
    if (error) {
      console.log(error(`error: ${error.message}`));
      return;
    }
    if (stderr) {
      console.log(error(`stderr: ${stderr}`));
      return;
    }

    console.log(`stdout: ${stdout}`);
  });

  // For now, redirect => should be changed in the future to direct to the next page
  console.log("reach here");
  res.redirect("/result");
};
