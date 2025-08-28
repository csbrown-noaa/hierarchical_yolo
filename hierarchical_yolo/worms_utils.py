import requests

WORMS_TREE_URL = 'https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{}'
WORMS_NAME_URL = 'https://www.marinespecies.org/rest/AphiaNameByAphiaID/{}'
WORMS_ID_URL = 'https://www.marinespecies.org/rest/AphiaIDByName/{}?marine_only=true&extant_only=true'

def get_WORMS_id(name):
    result = requests.get(WORMS_ID_URL.format(name))
    return int(result.content)

def get_WORMS_name(WORMS_id):
    result = requests.get(WORMS_NAME_URL.format(WORMS_id))
    return result.content.decode("utf-8")[1:-1]

def get_WORMS_tree(organism_id):
    '''
        >>> get_WORMS_tree('Gnathostomata')
        {
          "AphiaID": 1,
          "rank": "Superdomain",
          "scientificname": "Biota",
          "child": {
            "AphiaID": 2,
            "rank": "Kingdom",
            "scientificname": "Animalia",
            "child": {
              "AphiaID": 1821,
              "rank": "Phylum",
              "scientificname": "Chordata",
              "child": {
                "AphiaID": 146419,
                "rank": "Subphylum",
                "scientificname": "Vertebrata",
                "child": {
                  "AphiaID": 1828,
                  "rank": "Infraphylum",
                  "scientificname": "Gnathostomata",
                  "child": null
                }
              }
            }
          }
        }
    '''
    result = requests.get(WORMS_TREE_URL.format(organism_id))
    return result.json()

def WORMS_tree_to_childparent_tree(worms_trees):
    childparent_tree = {}
    for tree in worms_trees:
        try:
            parent = tree['AphiaID']
        except Exception as e:
            print("could not find id")
            print(tree)
            raise e
        while 'child' in tree and tree['child']:
            tree = tree['child']
            try:
                child = tree['AphiaID']
            except Exception as e:
                print("could not find id")
                print(tree)
                raise e
            childparent_tree[child] = parent
            parent = child
    return childparent_tree

def find_closest_permitted_parent(node, tree, permitted_nodes):
    if node not in tree:
        return None
    parent = tree[node]
    while parent not in permitted_nodes:
        if parent in tree:
            parent = tree[parent]
        else:
            return None
    return parent

def trim_childparent_tree(tree, permitted_nodes):
    new_tree = {}
    for node in tree:
        closest_permitted_parent = find_closest_permitted_parent(node, tree, permitted_nodes)
        new_tree[node] = closest_permitted_parent
    for node in list(new_tree.keys()):
        if new_tree[node] is None or (node not in permitted_nodes):
            del new_tree[node]
    return new_tree
        
def dict_keyvalue_replace(old_dict, replacemap):
    new_dict = {}
    for key in old_dict:
        new_dict[replacemap[key]] = replacemap[old_dict[key]]
    return new_dict
