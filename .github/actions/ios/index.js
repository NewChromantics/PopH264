const core = require("@actions/core");
const github = require("@actions/github");
const exec = require("@actions/exec");
const artifact = require("@actions/artifact");

const artifactClient = artifact.create();
const BuildScheme = core.getInput("BuildScheme");
const BuildProject = core.getInput("BuildProject");

async function run() {
  try {
    let regex = /TARGET_BUILD_DIR = [^\n]+\n/;
    const buildsettings = await exec.exec("xcodebuild", [
      `-workspace`, `${BuildProject}/project.xcworkspace`, `-scheme`, `${BuildScheme}`, `-showBuildSetting`,
    ]);
    console.log(buildsettings);
    const buildDirectory = regex.exec(buildsettings);
    console.log(buildDirectory);
    await exec.exec("xcodebuild", [
      `-workspace`, `${BuildProject}/project.xcworkspace`, `-scheme ${BuildScheme}`,
    ]);

    const files = ["PopH264_Ios.framework", "PopH264_Ios.framework.dSYM"];

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
