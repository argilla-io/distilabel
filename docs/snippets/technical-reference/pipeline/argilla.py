import argilla as rg

rg.init(api_key="<YOUR_ARGILLA_API_KEY>", api_url="<YOUR_ARGILLA_API_URL>")

rg_dataset = pipe_dataset.to_argilla()
rg_dataset.push_to_argilla(name="preference-dataset", workspace="admin")
