Write-Host $env:BUILDSOLUTION

set 
MSBuild $env:BUILDSOLUTION /property:Configuration=$env:BUILDCONFIGURATION /property:Platform=$env:BUILDPLATFORM

ls .