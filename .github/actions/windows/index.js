const core = require("@actions/core");
const exec = require("@actions/exec");

async function run() {
  try {
    process.env.BUILDSOLUTION = core.getInput("BuildSolution");
    process.env.BUILDPLATFORM = core.getInput("BuildPlatform");
    process.env.BUILDCONFIGURATION = core.getInput("BuildConfiguration");
    process.env.BUILDDIRECTORY = core.getInput("BuildDirectory");

    await exec.exec(`/bin/bash ${__dirname}/build.sh`);
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
