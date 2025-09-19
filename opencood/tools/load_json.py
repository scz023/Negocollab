import json

f = open('../../opencood/modality_assign/opv2v_4modality.json', 'r')
assi = json.load(f)
print(assi.items())