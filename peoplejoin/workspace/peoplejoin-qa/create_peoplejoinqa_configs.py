import os
import json
import random
import random

from data_preparation.spider.spider_common import AsyncCollabSpider, AsyncCollabTenantData, tenant_data_path_generator

random.seed(42)

workspace_path = "workspace/peoplejoin-qa/experiments"
dev_file_path = "data/peoplejoin-qa/dev.jsonl"
test_file_path = "data/peoplejoin-qa/test.jsonl"
exemplar_dialogue_ids = ["peoplejoinqa_1", "peoplejoinqa_2", "peoplejoinqa_3","peoplejoinqa_4"]
exemplar_dialogue_ids_nocot = ["peoplejoinqa_1_nocot", "peoplejoinqa_2_nocot", "peoplejoinqa_3_nocot","peoplejoinqa_4_nocot"]
exemplar_dialogue_ids_messageall = ["peoplejoinqa_messageall_1", "peoplejoinqa_messageall_2", "peoplejoinqa_messageall_3","peoplejoinqa_messageall_4"]
exemplar_dialogue_ids_messagenone = ["peoplejoinqa_messagenone_1", "peoplejoinqa_messagenone_2", "peoplejoinqa_messagenone_3","peoplejoinqa_messagenone_4"]

true = True
false = False

LLM_MODEL_CONFIGS = {
  "gpt-4o-2024-05-13": {
    "name": "dev-gpt-4o-2024-05-13",
    "context_window": 12000,
    "token_encoding": "o200k_base"
  },
  "gpt-4-turbo": {
      "name": "dev-gpt-4-turbo",
      "context_window": 30000
  },
  "phi-3-medium": {
      "name": "dev-phi-3-medium-128k-instruct",
      "context_window": 12000
  },
}

def get_agent_config_template(use_simple_cot: bool, 
                          dialogue_ids: list[str], 
                          llm_model: str = "gpt-4o-2024-05-13",
                          ):
  
  assert llm_model in LLM_MODEL_CONFIGS, f"llm_model should be one of {LLM_MODEL_CONFIGS.keys()}"
  plugin_ids = [
        "system",
        "enterprise",
        "enterprise_search"
  ]
  if use_simple_cot:
      plugin_ids.append("cot")
  agent_config_template = {
    "main_user_id": "alice",
    "tenant_id": "{tenant_id}",
    "model_config": {
      "model": LLM_MODEL_CONFIGS[llm_model]["name"], # only using the model name is sufficient
    },
    "exemplar_ids":dialogue_ids,
    "plugin_ids": plugin_ids
  }
  return agent_config_template

def get_experiment_template():
  experiment_template = {
    "tenant_id": "{tenant_id}",
    "participant_id_to_descriptions": 
    {
        "alice": "Alice Anastasiou is an employee at Company. She is interested in knowing {question} which may be available in her documents or notes or Agent might have to reach out to other people. When Agent initates interaction, start with '{question}'.",
        "lina": "Lina is an employee at Company. Lina will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "bhushan": "Bhushan Magar is an employee at Company. Bhushan will provide Bot with specific relevant information if it is available in his documents or notes, otherwise politely decline",
        "cassie": "Cassie Hicks is an employee at Company. Cassie will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "hannah": "Hannah Jarvis is an employee at Company. Hannah will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "dewei": "Dewei Peng is an employee at Company. Dewei will provide Bot with specific relevant information if it is available in his documents or notes, otherwise politely decline",
        "eden": "Eden is an employee at Company. Eden will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "parker": "Parker McLean is an employee at Company. Parker will provide Bot with specific relevant information if it is available in his documents or notes, otherwise politely decline",
        "farshid": "Farshid Kamangar is an employee at Company. Farshid will provide Bot with specific relevant information if it is available in his documents or notes, otherwise politely decline",
        "gorosti": "Gorosti Egiagarai is an employee at Company. Gorosti will provide Bot with specific relevant information if it is available in their documents or notes, otherwise politely decline",
        "harpreet": "Harpreet Thapar is an employee at Company. Harpreet will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "irena": "Irena Jovanovic is an employee at Company. Irena will provide Bot with specific relevant information if it is available in her documents or notes, otherwise politely decline",
        "juan": "Juan Quispe is an employee at Company. Juan will provide Bot with specific relevant information if it is available in his documents or notes, otherwise politely decline",
        "kerstin": "Kerstin Mark is an employee at Company. Kerstin will provide Bot with specific relevant information if it is available in her documents or notes",
        "maname": "Maname Mohlare is an employee at Company. Maname will provide Bot with specific relevant information if it is available in her documents or notes",
        "niks": "Niks Dzenis is an employee at Company. Niks will provide Bot with specific relevant information if it is available in her documents or notes",
        "oubunmi": "Oubunmi Gboyega is an employee at Company. Oubunmi will provide Bot with specific relevant information if it is available in her documents or notes",
        "ruwaidah": "Ruwaidah Fakhoury is an employee at Company. Ruwaidah will provide Bot with specific relevant information if it is available in her documents or notes",
        "sylvie": "Sylvie Rocher is an employee at Company. Sylvie will provide Bot with specific relevant information if it is available in her documents or notes",
        "tulga": "Tulga Bat-Erdene is an employee at Company. Tulga will provide Bot with specific relevant information if it is available in her documents or notes",
        "valarie": "Valarie Cabral is an employee at Company. Valarie will provide Bot with specific relevant information if it is available in her documents or notes"
    },
    "datum_id": "{datum_id}",
    "metadata": {
        "tenant_id": "{tenant_id}",
        "description": "Alice Anastasiou is an employee at Company. She is interested in knowing {question} which may be available in her documents or notes or Agent might have to reach out to other people.",
        "description_reference_response": "{reference_response}",
    },
    "agent_config_path": "{agent_config_path}",
    "default_instructions_file_name": "default_spider"
  }
  return experiment_template

def agent_config_path(tenant_id: str, llm_model: str, use_simple_cot: bool = false, messageall: bool = false, messagenone: bool = false) -> str:
    tenant_id_to_use = tenant_id.split("/")[-1]

    match (llm_model, use_simple_cot):
        case ("gpt-4o-2024-05-13", True):
            assert not (messageall or messagenone), f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall} and messagenone = {messagenone}"
            return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4o.json"
        case ("gpt-4o-2024-05-13", False):
            assert not (messageall or messagenone), f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall} and messagenone = {messagenone}"
            return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4o_nocot.json"
        case ("gpt-4-turbo", True):
            if messageall:
                assert not messagenone, f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall} and messagenone = {messagenone}"
                return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4turbo_messageall.json"
            elif messagenone:
                return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4turbo_messagenone.json"
            else:
                return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4turbo.json"   
        case ("gpt-4-turbo", False):
            if messageall or messagenone:
                raise ValueError(f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall}")
            else:
                return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_gpt4turbo_nocot.json"  
        case ("phi-3-medium", True):
            assert not (messageall or messagenone), f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall} and messagenone = {messagenone}"
            return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_phi3medium.json"   
        case ("phi-3-medium", False):
            assert not (messageall or messagenone), f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot} and messageall = {messageall} and messagenone = {messagenone}"
            return f"{workspace_path}/agent_configs/agentconf_{tenant_id_to_use}_phi3medium_nocot.json"                        
        case (_, _):
            raise ValueError(f"Invalid llm_model = {llm_model} and use_simple_cot = {use_simple_cot}")

def get_reference_people(async_collab_tenant_data_id:str, datum:AsyncCollabSpider) -> list[str]:
    # print("Getting reference people for tenant_id = ", async_collab_tenant_data_id, " and datum_id = ", datum.datum_id)
    async_collab_tenant_data_file = tenant_data_path_generator(
        async_collab_tenant_data_id
    )
    tenant = AsyncCollabTenantData.load_from_file(async_collab_tenant_data_file)
    reference_people = datum.get_reference_people_excluding_primary(tenant)
    return list(reference_people)


def create_experiment_file(datum: AsyncCollabSpider, experiment_file_path: str, primary_user_id: str = "alice", use_simple_cot: bool = false, llm_model: str = "gpt-4o-2024-05-13", messageall: bool = false, messagenone: bool = false):
    print("Creating experiment file for datum_id = ", datum.datum_id)
    print("datum.tenant_id = ", datum.tenant_id)
    experiment_template_copy = get_experiment_template()
    experiment_template_copy["datum_id"] = datum.datum_id
    experiment_template_copy["agent_config_path"] = agent_config_path(tenant_id=datum.tenant_id, llm_model=llm_model, use_simple_cot=use_simple_cot, messageall=messageall, messagenone=messagenone)
    # also replace {question} in participant_id_to_descriptions
    experiment_template_copy["participant_id_to_descriptions"][primary_user_id] = experiment_template_copy["participant_id_to_descriptions"][primary_user_id].format(question=datum.question)
    experiment_template_copy["metadata"]["description"] = experiment_template_copy["metadata"]["description"].format(
        question=datum.question, reference_response=datum.execution_result
    )
    experiment_template_copy["metadata"]["description_reference_response"] = datum.execution_result
    experiment_template_copy["metadata"]["description_reference_people"] = json.dumps(get_reference_people(datum.tenant_id, datum))
    experiment_template_copy["metadata"]["tenant_id"] = experiment_template_copy["metadata"]["description"].format(
        tenant_id=datum.tenant_id
    )
    experiment_template_copy["tenant_id"] = experiment_template_copy["tenant_id"].format(tenant_id=datum.tenant_id)
    print(f"Saving experiment to {experiment_file_path}")
    with open(experiment_file_path, "w") as f:
        json.dump(experiment_template_copy, f, indent=4)

def create_agent_config_file(tenant_id: str, agent_config_file_path: str, use_simple_cot: bool = false, llm_model: str = "gpt-4o-2024-05-13", messageall: bool = false, messagenone: bool = false):
    print("Creating agent config file for tenant_id = ", tenant_id)
    dialogue_ids = exemplar_dialogue_ids if use_simple_cot else exemplar_dialogue_ids_nocot
    if messageall:
        assert use_simple_cot, f"messageall is only supported for use_simple_cot = true"
        dialogue_ids = exemplar_dialogue_ids_messageall
    elif messagenone:
        assert use_simple_cot, f"messagenone is only supported for use_simple_cot = true"
        dialogue_ids = exemplar_dialogue_ids_messagenone
    agent_config_template = get_agent_config_template(use_simple_cot=use_simple_cot, 
                                                      dialogue_ids=dialogue_ids,
                                                      llm_model=llm_model)
    agent_config_template["tenant_id"] = f"{tenant_id}"
    print(f"Saving agent config to {agent_config_file_path}")
    with open(agent_config_file_path, "w") as f:
        json.dump(agent_config_template, f, indent=4)


def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev", help="Split to use for creating configs")
    args = parser.parse_args()

    exp_configs_folder = "exp_configs"

    if args.split == "dev":
        file_path = dev_file_path
    elif args.split == "test":
        file_path = test_file_path
        exp_configs_folder = "exp_configs_test"
    else:
        print(f"Error: Invalid split = {args.split}. Should be one of ['dev', 'test']")
        return
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Error: {file_path} does not exist or is empty.")
        return
        
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    
    random.shuffle(data)

    # we will skip some tenants with very long documents, as they will not fit simulated user's context window
    # maxdoclenth, tenantname, tenant_id_count_across_data
    # 152975128 tenant_data_v2/soccer_1 14
    # 12765927 tenant_data_v2/flight_4 82
    # 3175902 tenant_data_v2/college_2 170
    # 3101304 tenant_data_v2/bike_1 104
    # 668158 tenant_data_v2/store_1 112
    # 105948 tenant_data_v2/wine_1 82
    # 82719 tenant_data_v2/csu_1 70
    tenants_to_skip = {"tenant_data_v2/soccer_1", "tenant_data_v2/flight_4", "tenant_data_v2/college_2", "tenant_data_v2/bike_1", "tenant_data_v2/store_1", "tenant_data_v2/wine_1", "tenant_data_v2/csu_1"}

    # filter
    data = [datum for datum in data if datum["tenant_id"] not in tenants_to_skip]

    # get first 500 
    data = data[:500]

    selected_data = [AsyncCollabSpider.from_dict(datum) for datum in data]

    # print the datum and tenant id of all the selected data to f"{workspace_path}/selected_data.txt"
    with open(f"{workspace_path}/selected_data.txt", "w") as f:
        f.write("datum_id, tenant_id\n")
        for datum in selected_data:
            if hasattr(datum, 'datum_id') and hasattr(datum, 'tenant_id'):
                f.write(f"{datum.datum_id}, {datum.tenant_id}\n")
            else:
                print(f"Warning: datum {datum} is missing datum_id or tenant_id")    

    # create "{workspace_path}/agent_configs" directory if it does not exist
    if not os.path.exists(f"{workspace_path}/agent_configs"):
        os.makedirs(f"{workspace_path}/agent_configs")

    for use_simple_cot in [false, true]:
      agent_file_tracker = {}
      print("use_simple_cot = ", use_simple_cot)
      exp_file_dir = f"{workspace_path}/gpt4o/{exp_configs_folder}" if use_simple_cot else f"{workspace_path}/gpt4o_nocot/{exp_configs_folder}"
      # create exp_file_dir if it does not exist
      if not os.path.exists(exp_file_dir):
          os.makedirs(exp_file_dir)
      for i, datum in enumerate(selected_data):
          print("datum = ", datum)
          create_experiment_file(datum, f"{exp_file_dir}/experiment_{i}.json", use_simple_cot=use_simple_cot)
          # create agent config file if not already created
          if datum.tenant_id not in agent_file_tracker:
              create_agent_config_file(datum.tenant_id, agent_config_path(tenant_id=datum.tenant_id, use_simple_cot=use_simple_cot, llm_model="gpt-4o-2024-05-13"), use_simple_cot=use_simple_cot, llm_model="gpt-4o-2024-05-13")
              agent_file_tracker[datum.tenant_id] = True
          print()

    for use_simple_cot in [false, true]:
      agent_file_tracker = {}
      print("use_simple_cot = ", use_simple_cot)
      exp_file_dir = f"{workspace_path}/gpt4turbo/{exp_configs_folder}" if use_simple_cot else f"{workspace_path}/gpt4turbo_nocot/{exp_configs_folder}"
      # create exp_file_dir if it does not exist
      if not os.path.exists(exp_file_dir):
          os.makedirs(exp_file_dir)
      for i, datum in enumerate(selected_data):
          print("datum = ", datum)
          create_experiment_file(datum, f"{exp_file_dir}/experiment_{i}.json", use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo")
          # create agent config file if not already created
          if datum.tenant_id not in agent_file_tracker:
              agent_config_path_to_use = agent_config_path(tenant_id=datum.tenant_id, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo")
              create_agent_config_file(datum.tenant_id, agent_config_path_to_use, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo")
              agent_file_tracker[datum.tenant_id] = True
          print()   
    

    # gpt-4-turbo alternative configs
    # _messageall
    for use_simple_cot in [true]:
        agent_file_tracker = {}
        print("use_simple_cot = ", use_simple_cot)
        exp_file_dir = f"{workspace_path}/gpt4turbo_messageall/{exp_configs_folder}" if use_simple_cot else f"{workspace_path}/gpt4turbo_messageall_nocot/{exp_configs_folder}"
        # create exp_file_dir if it does not exist
        if not os.path.exists(exp_file_dir):
            os.makedirs(exp_file_dir)
        for i, datum in enumerate(selected_data):
            print("datum = ", datum)
            create_experiment_file(datum, f"{exp_file_dir}/experiment_{i}.json", use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messageall=True)
            # create agent config file if not already created
            if datum.tenant_id not in agent_file_tracker:
                agent_config_path_to_use = agent_config_path(tenant_id=datum.tenant_id, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messageall=True)
                create_agent_config_file(datum.tenant_id, agent_config_path_to_use, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messageall=True)
                agent_file_tracker[datum.tenant_id] = True
            print() 

    # _messagenone
    for use_simple_cot in [true]:
        agent_file_tracker = {}
        print("use_simple_cot = ", use_simple_cot)
        exp_file_dir = f"{workspace_path}/gpt4turbo_messagenone/{exp_configs_folder}" if use_simple_cot else f"{workspace_path}/gpt4turbo_messagenone_nocot/{exp_configs_folder}"
        # create exp_file_dir if it does not exist
        if not os.path.exists(exp_file_dir):
            os.makedirs(exp_file_dir)
        for i, datum in enumerate(selected_data):
            print("datum = ", datum)
            create_experiment_file(datum, f"{exp_file_dir}/experiment_{i}.json", use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messagenone=True)
            # create agent config file if not already created
            if datum.tenant_id not in agent_file_tracker:
                agent_config_path_to_use = agent_config_path(tenant_id=datum.tenant_id, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messagenone=True)
                create_agent_config_file(datum.tenant_id, agent_config_path_to_use, use_simple_cot=use_simple_cot, llm_model="gpt-4-turbo", messagenone=True)
                agent_file_tracker[datum.tenant_id] = True
            print()               
    

    # # create agent config files with phi-3-medium
    for use_simple_cot in [false, true]:
      agent_file_tracker = {}
      print("use_simple_cot = ", use_simple_cot)
      exp_file_dir = f"{workspace_path}/phi3medium/{exp_configs_folder}" if use_simple_cot else f"{workspace_path}/phi3medium_nocot/{exp_configs_folder}"
      # create exp_file_dir if it does not exist
      if not os.path.exists(exp_file_dir):
          os.makedirs(exp_file_dir)
      for i, datum in enumerate(selected_data):
          print("datum = ", datum)
          create_experiment_file(datum, f"{exp_file_dir}/experiment_{i}.json", use_simple_cot=use_simple_cot, llm_model="phi-3-medium")
          # create agent config file if not already created
          if datum.tenant_id not in agent_file_tracker:
              create_agent_config_file(datum.tenant_id, agent_config_path(tenant_id=datum.tenant_id, use_simple_cot=use_simple_cot, llm_model="phi-3-medium"), use_simple_cot=use_simple_cot, llm_model="phi-3-medium")
              agent_file_tracker[datum.tenant_id] = True
          print()
  


if __name__ == "__main__":
    main()


# python workspace/peoplejoin-qa/create_peoplejoinqa_configs.py --split test