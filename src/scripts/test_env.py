from src.lab.client import load_config, get_client

cfg = load_config()
print(cfg)

client = get_client()
print("Client ok:", type(client))
