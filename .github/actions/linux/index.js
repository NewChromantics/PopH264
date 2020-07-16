const core = require("@actions/core");
const github = require("@actions/github");
const exec = require("@actions/exec");
const artifact = require("@actions/artifact");

const artifactClient = artifact.create();
const makefile = core.getInput("makefile");
const architecture = core.getInput("architecture");

async function run() {
  try {
    process.env.BUILDSOLUTION = architecture;
    if(makefile === 'Makefile') {
    await exec.exec("sudo", [
      `add-apt-repository -y ppa:ubuntu-toolchain-r/test`,
    ]);
    await exec.exec("sh", [`sudo apt-get update`]);
    await exec.exec("sudo", [
      `apt-get install libx264-dev gcc-10 g++-10 -y`,
    ]);
    await exec.exec("sudo", [
      `update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10`,
    ]);
    await exec.exec("sudo", [
      `update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10`,
    ]);
  }

    await exec.exec("make", [
      `-f ${makefile}`, `GithubWorkflow`, `-C PopH264.Linux/`,
    ]);

    const files = [
      `PopH264${architecture}.so`,
      `PopH264TestApp${architecture}`,
    ];

    const buildDirectory = "./PopH264.Linux";

    const options = {
      continueOnError: false,
    };
    const uploadResponse = await artifactClient.uploadArtifact(
      artifactName,
      files,
      buildDirectory,
      options
    );
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
