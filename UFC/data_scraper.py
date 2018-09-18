import os
import requests
import bs4
import pandas as pd

def get_fighter_data():
	df = pd.DataFrame(columns = fighter_cols)
    
	for page_key in page_keys:
		page_res = requests.get("http://www.fightmetric.com/statistics/fighters?char=" + page_key + "&page=all")
		page_soup = bs4.BeautifulSoup(page_res.text, "lxml")
		fighter_links = []
		
		for i in page_soup.select(".b-link.b-link_style_black"):
			fighter_links.append(i.get("href"))
		
		fighter_links = list(set(fighter_links))
		
		for i in fighter_links:
			fighter_res = requests.get(i)
			fighter_soup = bs4.BeautifulSoup(fighter_res.text, "lxml")
			row = [fighter_soup.select(".b-content__title-highlight")[0].text.strip()]
			fighter_dat = fighter_soup.select(".b-list__box-list-item.b-list__box-list-item_type_block")
			
			for j in fighter_dat:
				if (j.text.strip() == ""):
					continue
				j.i.decompose()
				row.append(j.text.strip())
			
			df = df.append(pd.DataFrame([row], columns = fighter_cols))

	return df

def get_fight_data(out_file_name):
	df = pd.DataFrame(columns = fight_cols)
	
	page_res = requests.get("http://www.fightmetric.com/statistics/events/completed?page=all")
	page_soup = bs4.BeautifulSoup(page_res.text, "lxml")
	event_links = []
	
	for i in page_soup.select(".b-link.b-link_style_black"):
	    print(i.get("href"))
	    event_res = requests.get(i.get("href"))
	    event_soup = bs4.BeautifulSoup(event_res.text, "lxml")
	
	    for j in event_soup.select(".b-fight-details__table-row.b-fight-details__table-row__hover.js-fight-details-click"):
	        data = j.select(".b-fight-details__table-text")
	        row = [data[1].text.strip(),
	               data[2].text.strip(),
	               data[1].text.strip(),
	               data[11].text.strip()]
	        df = df.append(pd.DataFrame([row], columns = fight_cols))
	
	return df

page_keys = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
fighter_cols = ["NAME", "Height", "Weight", "REACH", "Stance", "DOB", "SLPM", "STRA", "SAPM", "STRD", "TD", "TDA", "TDD", "SUBA"]
fight_cols = ["Fighter1", "Fighter2", "Winner", "WeightClass"]

fighter_data = get_fighter_data()
fight_data = get_fight_data()

fighter_data.to_csv(os.path.join(os.path.dirname(__file__), "data/fighters.csv"))
fight_data.to_csv(os.path.join(os.path.dirname(__file__), "data/fights.csv"))
