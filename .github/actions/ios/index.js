const core = require("@actions/core");
const github = require("@actions/github");
const exec = require("@actions/exec");
const artifact = require("@actions/artifact");

const BuildScheme = core.getInput("BuildScheme");
const BuildProject = core.getInput("BuildProject");

const artifactClient = artifact.create();
const artifactName = BuildScheme;

const regex = /TARGET_BUILD_DIR = ([\/-\w]+)\n/;
let buildDirectory = "";
let myError = "";

async function run() {
  try {
    const outputOptions = {};
    outputOptions.listeners = {
      stdout: (data) => {
        buildDirectory += data.toString();
        console.log(buildDirectory.indexOf("TARGET_BUILD_DIR ="));
        console.log(regex.exec(buildDirectory));
        buildDirectory = regex.exec(buildDirectory);
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
      ],
      outputOptions
    );

    console.log(buildDirectory);
    await exec.exec("xcodebuild", [
      `-workspace`,
      `${BuildProject}/project.xcworkspace`,
      `-scheme`,
      `${BuildScheme}`,
    ]);

    const files = [
      `${buildDirectory}/PopH264_Ios.framework`,
      `${buildDirectory}/PopH264_Ios.framework.dSYM`,
    ];

    const rootDirectory = buildDirectory[1];

    console.log(buildDirectory[1]);

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
