import numpy as np
import json


def one_hot_encode(labels):
    """
    Convert a list of integer labels to one-hot encoded format.

    Parameters:
    labels (list or np.array): List or array of integer labels.
    num_classes (int): Total number of classes.

    Returns:
    np.array: One-hot encoded representation of the input labels.
    """
    labels = np.array(labels)
    unique_labels, invese_indices = np.unique(labels, return_inverse=True)
    one_hot = np.eye(len(unique_labels))[invese_indices]
    return one_hot


def load_jsonl(filepath):
    """Load a .jsonl file as a list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def encode_jsonl_to_one_hot(data, key):
    """Encode a specific key in a list of dicts to one-hot format."""
    labels = [item[key] for item in data]
    one_hot_encoded = one_hot_encode(labels)
    return one_hot_encoded


if __name__ == "__main__":
    agent_file = "KnutKnut/KnutKnutRealEstate/data/agents.jsonl"
    district_file = "KnutKnut/KnutKnutRealEstate/data/districts.jsonl"
    house_file = "KnutKnut/KnutKnutRealEstate/data/houses.jsonl"
    school_file = "KnutKnut/KnutKnutRealEstate/data/schools.jsonl"

    agents = load_jsonl(agent_file)
    districts = load_jsonl(district_file)
    houses = load_jsonl(house_file)
    schools = load_jsonl(school_file)

    agents_dict = {agent['agent_id']: agent for agent in agents}
    districts_dict = {district['id']: district for district in districts}
    schools_dict = {school['id']: school for school in schools}

    for house in houses:
        # Replace agent_id with agent_name, if available
        agent_info = agents_dict.get(house.get('agent_id'))
        if agent_info is not None:
            house['agent_name'] = agent_info.get('name')
        house.pop('agent_id', None)

        # Replace district_id with district_name, if available
        district_info = districts_dict.get(house.get('district_id'))
        if district_info is not None:
            house['district_crime_rating'] = district_info.get('crime_rating')
            house['district_public_transport_rating'] = district_info.get(
                'public_transport_rating')
        house.pop('district_id', None)

        # Replace school_id with school_name, if available
        school_info = schools_dict.get(house.get('school_id'))
        if school_info is not None:
            house['school_built_year'] = school_info.get('built_year')
            house['school_rating'] = school_info.get('rating')
            house['school_capacity'] = school_info.get('capacity')
        house.pop('school_id', None)

    # Print the first house with enriched data for verification
    # print(houses[0])

    print(houses[0].keys())

    for key, value in houses[0].items():
        if type(value) == str:
            one_hot = encode_jsonl_to_one_hot(houses, key)
            print(f"One-hot encoding for {key}:")
            print(one_hot)


print(3 == float('3'))  # True
