const core = require('@actions/core');
const exec = require("@actions/exec");

try {
  process.env.BUILDPROJECT = core.getInput("BuildProject");
  process.env.BUILDSCHEME = core.getInput("BuildScheme");
  
  await exec.exec(`/bin/bash ${__dirname}/build.sh`);

} catch (error) {
  core.setFailed(error.message);
}
