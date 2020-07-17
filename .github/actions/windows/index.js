const core = require("@actions/core");
const exec = require("@actions/exec");

async function run() {
  try {
    const BUILDSOLUTION = core.getInput("BuildSolution");
    const BUILDPLATFORM = core.getInput("BuildPlatform");
    const BUILDCONFIGURATION = core.getInput("BuildConfiguration");
    const BUILDDIRECTORY = core.getInput("BuildDirectory");

    await exec.exec(`powershell.exe ${__dirname}/build.ps1`);

    await exec.exec('Write-Host', [BUILDSOLUTION])
    await exec.exec('set')
    await exec.exec('MSBuild',[BUILDSOLUTION, `property:Configuration=${BUILDCONFIGURATION}`, `property:Platform=${BUILDPLATFORM}`])
    await exec.exec('ls', [BUILDDIRECTORY])

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
