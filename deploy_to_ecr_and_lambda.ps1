<#
Deploy script: builds Docker image, pushes to ECR, and creates/updates Lambda (image).
Usage:
  PowerShell (from this script folder):
    .\deploy_to_ecr_and_lambda.ps1 -Region us-east-1 -RepoName my-car-damage-repo
Requirements:
  - Docker running locally
  - AWS CLI configured with permissions for ECR, Lambda
#>
param(
  [string]$Region = "us-east-1",
  [string]$RepoName = "my-car-damage-repo",
  [string]$ImageTag = "latest",
  [string]$LambdaFunctionName = "car-damage-detector",
  [string]$RoleArn = ""
)

Write-Host "Starting deployment..." -ForegroundColor Cyan
Write-Host "Region: $Region" -ForegroundColor Cyan
Write-Host "RepoName: $RepoName" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

Write-Host "Checking ECR repository..." -ForegroundColor Yellow
$repoUri = aws ecr describe-repositories --repository-names $RepoName --region $Region --query "repositories[0].repositoryUri" --output text 2>$null
if (!$repoUri -or $repoUri -eq "None") {
  Write-Host "ECR repository not found; creating $RepoName..." -ForegroundColor Yellow
  $create = aws ecr create-repository --repository-name $RepoName --region $Region --output json
  $repoUri = ($create | ConvertFrom-Json).repository.repositoryUri
}
Write-Host "Repository URI: $repoUri" -ForegroundColor Green

$registry = $repoUri.Split('/')[0]

Write-Host "Logging into ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { 
  Write-Host "Docker login failed" -ForegroundColor Red
  exit 1 
}

$imageName = "$RepoName`:$ImageTag"
Write-Host "Building Docker image: $imageName" -ForegroundColor Yellow
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) { 
  Write-Host "Docker build failed" -ForegroundColor Red
  exit 1 
}

$imageUri = "$repoUri`:$ImageTag"
Write-Host "Tagging image as: $imageUri" -ForegroundColor Yellow
docker tag $imageName $imageUri

Write-Host "Pushing image to ECR..." -ForegroundColor Yellow
docker push $imageUri
if ($LASTEXITCODE -ne 0) { 
  Write-Host "Docker push failed" -ForegroundColor Red
  exit 1 
}

Write-Host "Checking for Lambda function..." -ForegroundColor Yellow
aws lambda get-function --function-name $LambdaFunctionName --region $Region > $null 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Host "Updating Lambda function code..." -ForegroundColor Yellow
  aws lambda update-function-code --function-name $LambdaFunctionName --image-uri $imageUri --region $Region
  if ($LASTEXITCODE -ne 0) { 
    Write-Host "Update failed" -ForegroundColor Red
    exit 1 
  }
  Write-Host "Updated successfully." -ForegroundColor Green
} else {
  if (-not $RoleArn) {
    Write-Host "Lambda not found and no RoleArn provided." -ForegroundColor Red
    exit 1
  }
  Write-Host "Creating Lambda function..." -ForegroundColor Yellow
  aws lambda create-function --function-name $LambdaFunctionName --package-type Image --code ImageUri=$imageUri --role $RoleArn --memory-size 4096 --timeout 120 --region $Region
  if ($LASTEXITCODE -ne 0) { 
    Write-Host "Creation failed" -ForegroundColor Red
    exit 1 
  }
  Write-Host "Created successfully." -ForegroundColor Green
}

Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host "Image URI: $imageUri" -ForegroundColor Green