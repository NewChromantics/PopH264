const core = require("@actions/core");
const exec = require("@actions/exec");

const BUILDPLATFORM = core.getInput("BuildPlatform");
const BUILDCONFIGURATION = core.getInput("BuildConfiguration");
const BUILDDIRECTORY = core.getInput("BuildDirectory");
const PROJECT = core.getInput("Project")

const BuildSolution = `${PROJECT}.visualstudio/${PROJECT}.sln`

async function run() {
  try {


    await exec.exec(`powershell.exe ${__dirname}/build.ps1`);

    await exec.exec('Write-Host', [BuildSolution])
    await exec.exec('set')
    await exec.exec('MSBuild',[BuildSolution, `property:Configuration=${BUILDCONFIGURATION}`, `property:Platform=${BUILDPLATFORM}`])
    await exec.exec('ls', [BUILDDIRECTORY])

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
