# Define variables
$anacondaFile = "Anaconda3-2024.06-1-Windows-x86_64.exe"
$carlaVersion = "0.9.15"
$carlaUrl = "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_${carlaVersion}.tar.gz"
$carlaDir = "C:\carla-simulator"
$githubRepo = "https://github.com/ENDEAVR-E2E-Autonomous-Driving/DDPG-IL"

# Download and install Anaconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/archive/$anacondaFile" -OutFile $anacondaFile
Start-Process -Wait -FilePath .\$anacondaFile -ArgumentList "/S /D=C:\Anaconda3"

# Add Anaconda to PATH
$env:Path += ";C:\Anaconda3;C:\Anaconda3\Scripts;C:\Anaconda3\Library\bin"

# Download CARLA
Invoke-WebRequest -Uri $carlaUrl -OutFile "CARLA_${carlaVersion}.tar.gz"

# Unpack CARLA
New-Item -ItemType Directory -Force -Path $carlaDir
tar -xzvf "CARLA_${carlaVersion}.tar.gz" -C $carlaDir

# Install CARLA Python module and dependencies
# pip install carla==$carlaVersion
# pip install -r "$carlaDir\PythonAPI\examples\requirements.txt"

# Clone the GitHub repository
git clone $githubRepo

# Clean up
Remove-Item -Force -Path .\$anacondaFile, "CARLA_${carlaVersion}.tar.gz"

Write-Host "Installation completed successfully!"
