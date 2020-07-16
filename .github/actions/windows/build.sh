#!/bin/bash -e

set MSBuild %BUILDSOLUTION% /property:Configuration=%BUILDCONFIGURATION% /property:Platform=%BUILDPLATFORM%

ls 