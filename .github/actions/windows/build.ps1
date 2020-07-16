Write-Host $BUILDSOLUTION

set MSBuild $BUILDSOLUTION /property:Configuration=$BUILDCONFIGURATION /property:Platform=$BUILDPLATFORM

ls .