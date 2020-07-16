const core = require("@actions/core");
const exec = require("@actions/exec");

async function run() {
  try {
    process.env.ARCHITECTURE = core.getInput("architecture");
    process.env.MAKEFILE = core.getInput("makefile");

    await exec.exec(`/bin/bash ${__dirname}/build.sh`);
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
