const core = require("@actions/core");
const github = require("@actions/github");
const exec = require("@actions/exec");
const artifact = require("@actions/artifact");

const architecture = core.getInput("architecture");

const artifactClient = artifact.create();
const artifactName = `linux-${architecture}`;

const project = core.getInput("project").slice(14)

async function run() {
  try {
    console.log(await exec.exec("ls"));
    console.log(project)
    process.env.archTarget = architecture;
    console.log('echo', ['$GITHUB_REPOSITORY'])
    // For Gihub hosted runners need to update gcc and get libx264
    if (architecture === "x86_64") {
      await exec.exec("sudo", [
        `add-apt-repository`,
        `-y`,
        `ppa:ubuntu-toolchain-r/test`,
      ]);
      await exec.exec("sudo", [`apt-get`, `update`]);
      await exec.exec("sudo", [
        `apt-get`,
        `install`,
        `libx264-dev`,
        `gcc-10`,
        `g++-10`,
        `-y`,
      ]);
      await exec.exec("sudo", [
        `update-alternatives`,
        `--install`,
        `/usr/bin/gcc`,
        `gcc`,
        `/usr/bin/gcc-10`,
        `10`,
      ]);
      await exec.exec("sudo", [
        `update-alternatives`,
        `--install`,
        `/usr/bin/g++`,
        `g++`,
        `/usr/bin/g++-10`,
        `10`,
      ]);
    }

    await exec.exec("make", [`exec`, `-C`, `PopH264.Linux/`]);

    // These can be parameters passed in at a later state
    const files = [
      `Build/Linux_${architecture}/libPopH264.so`,
      `Build/Linux_${architecture}/PopH264TestApp`,
      `Build/Linux_${architecture}/PopH264.h`,
    ];

    const rootDirectory = ".";

    const options = {
      continueOnError: false,
    };
    const uploadResponse = await artifactClient.uploadArtifact(
      artifactName,
      files,
      rootDirectory,
      options
    );
  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
