from picker_model import TargetModel
from collect_data import collect_links
import json

def get_links(car_brand="toyota", max_pages=3, max_links=15, offset=0):
    with open('/content/part-number-recognition/prompts.json', 'r') as f:
      prompts = json.load(f)

    first_page_link = prompts[car_brand.lower()]['first_page_url']

    t = TargetModel()
    links = collect_links(t, first_page_link, max_pages=max_pages, max_links=max_links, offset=offset)
    print("Number of links received: ",len(links))
    return links
