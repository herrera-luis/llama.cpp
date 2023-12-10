module "chat_service" {

  source = "git::https://github.com/herrera-luis/tf-modules.git//ecs-service?ref=v0.0.1"
  #source = "../../tf-modules/ecs-service"

  name                = "llm-chat"
  vpc_id              = data.terraform_remote_state.vpc.outputs.us_west_2.vpc_id
  ecs_cluster_id      = data.terraform_remote_state.llama_cpp_service.outputs.ecs_cluster_id
  private_subnet_ids  = data.terraform_remote_state.vpc.outputs.us_west_2.private_subnet_ids
  efs_system_id       = data.terraform_remote_state.llama_cpp_service.outputs.efs_id
  lb_target_group_arn = data.terraform_remote_state.vpc.outputs.us_west_2.lb_chat_target_group_arn
  sg_ingress_ports = {
    from_port = 9000
    to_port   = 9000
  }
  service_discovery = {
    enabled      = true
    namespace_id = data.terraform_remote_state.llama_cpp_service.outputs.service_discovery_private_dns_namespace_id
  }
  container = {
    name          = "chat"
    desired_count = 1
    cpu           = "8192"
    memory        = "16384"
    portMappings = {
      containerPort = 9000
      hostPort      = 9000
    }
    command = ["--server", "-m", "models/llama-2-7b-chat.Q5_0.gguf", "-c", "4096", "--threads", "6", "--port", "9000", "--host", "0.0.0.0"]
  }
  tag_version = var.tag_version
}


module "image_service" {
  source = "git::https://github.com/herrera-luis/tf-modules.git//ecs-service?ref=v0.0.1"
  #source = "../../tf-modules/ecs-service"

  name                = "llm-image"
  vpc_id              = data.terraform_remote_state.vpc.outputs.us_west_2.vpc_id
  ecs_cluster_id      = data.terraform_remote_state.llama_cpp_service.outputs.ecs_cluster_id
  private_subnet_ids  = data.terraform_remote_state.vpc.outputs.us_west_2.private_subnet_ids
  efs_system_id       = data.terraform_remote_state.llama_cpp_service.outputs.efs_id
  lb_target_group_arn = data.terraform_remote_state.vpc.outputs.us_west_2.lb_image_target_group_arn
  sg_ingress_ports = {
    from_port = 8080
    to_port   = 8080
  }

  service_discovery = {
    enabled      = true
    namespace_id = data.terraform_remote_state.llama_cpp_service.outputs.service_discovery_private_dns_namespace_id
  }

  container = {
    name          = "image"
    desired_count = 1
    cpu           = "8192"
    memory        = "16384"
    portMappings = {
      containerPort = 8080
      hostPort      = 8080
    }
    command = ["--server", "-m", "models/ggml-model-f16.gguf", "--mmproj", "models/mmproj-model-f16.gguf", "--threads", "6", "--port", "8080", "-ngl", "1", "--host", "0.0.0.0"]

  }
  tag_version = var.tag_version

}
