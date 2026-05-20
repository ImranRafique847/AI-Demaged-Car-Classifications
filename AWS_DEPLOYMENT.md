# AWS Lambda Deployment Guide
## Car Damage Detection System

### Prerequisites
- AWS Account
- AWS CLI installed: `pip install awscli`
- Docker installed
- AWS CLI configured: `aws configure`

---

## Step 1 — Configure AWS CLI

```bash
aws configure
```
Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Region: `us-east-1` (or your preferred region)
- Output format: `json`

---

## Step 2 — Create ECR Repository

```bash
aws ecr create-repository --repository-name car-damage-detection --region us-east-1
```

Note the `repositoryUri` from the output. It looks like:
`123456789.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection`

---

## Step 3 — Build & Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t car-damage-detection .

# Tag image
docker tag car-damage-detection:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection:latest

# Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection:latest
```

Alternatively, use the provided PowerShell deployment script from the repository root:

```powershell
.\deploy_to_ecr_and_lambda.ps1 -Region us-east-1 -RepoName car-damage-detection -LambdaFunctionName car-damage-detection -RoleArn arn:aws:iam::123456789:role/lambda-execution-role
```

---

## Step 4 — Create Lambda Function

```bash
aws lambda create-function \
  --function-name car-damage-detection \
  --package-type Image \
  --code ImageUri=123456789.dkr.ecr.us-east-1.amazonaws.com/car-damage-detection:latest \
  --role arn:aws:iam::123456789:role/lambda-execution-role \
  --memory-size 3008 \
  --timeout 60 \
  --region us-east-1
```

---

## Step 5 — Create API Gateway

```bash
# Create HTTP API
aws apigatewayv2 create-api \
  --name car-damage-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:123456789:function:car-damage-detection
```

Your app will be available at the API Gateway URL.

---

## Lambda Settings
| Setting | Value |
|---------|-------|
| Memory | 3008 MB |
| Timeout | 60 seconds |
| Architecture | x86_64 |

---

## Cost Estimate
- First 1M requests/month: **FREE**
- After that: ~$0.20 per 1M requests
- For a student project: **essentially free**
