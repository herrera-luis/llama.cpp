terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=5.6.0, <=5.7.0"
    }
  }
  backend "s3" {
    bucket = "dev-llama-cpp-state"
    key    = "llama-cpp.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = "us-west-2"
  alias  = "us_west_2"
}

data "terraform_remote_state" "vpc" {
  backend = "s3"
  config = {
    bucket = "${local.environment}-llm-infrastructure-base"
    key    = "vpc-llm-infrastructure-terraform.tfstate"
    region = "us-west-2"
  }
}


data "terraform_remote_state" "llama_cpp_service" {
  backend = "s3"
  config = {
    bucket = "${local.environment}-llm-infrastructure-base"
    key    = "llama-cpp.tfstate"
    region = "us-west-2"
  }
}

locals {
  environment = "dev"
}
