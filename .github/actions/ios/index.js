const core = require("@actions/core");
const github = require("@actions/github");
const exec = require("@actions/exec");
const artifact = require("@actions/artifact");

const BuildScheme = core.getInput("BuildScheme");
const BuildProject = core.getInput("BuildProject");

const artifactClient = artifact.create();
const artifactName = BuildScheme;

const regex = /TARGET_BUILD_DIR = [^\n]+\n/;
let myOutput = "";
let myError = "";

async function run() {
  try {
    const outputOptions = {};
    outputOptions.listeners = {
      stdout: (data) => {
        myOutput = data.toString();
        console.log(typeof myOutput, myOutput);
      },
      stderr: (data) => {
        myError += data.toString();
      },
    };

    const buildsettings = await exec.exec(
      "xcodebuild",
      [
        `-workspace`,
        `${BuildProject}/project.xcworkspace`,
        `-scheme`,
        `${BuildScheme}`,
        `-showBuildSettings`,
        `|`,
        `grep TARGET_BUILD_DIR`,
        `|`,
        `sed`,
        `-e`,
        `s/.*TARGET_BUILD_DIR = //`,
      ],
      outputOptions
    );

    console.log(buildsettings);
    const buildDirectory = regex.exec(buildsettings);
    console.log(buildDirectory);
    await exec.exec("xcodebuild", [
      `-workspace`,
      `${BuildProject}/project.xcworkspace`,
      `-scheme`,
      `${BuildScheme}`,
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
