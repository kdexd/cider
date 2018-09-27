# demo script for running CIDEr
import json
import os

from pyciderevalcap.eval import CIDErEvalCap as ciderEval

# load the configuration file
config = json.loads(open('params.json', 'r').read())

pathToData = config['pathToData']
refName = config['refName']
candName = config['candName']
resultFile = config['resultFile']
df_mode = config['idf']

# Print the parameters
print("Running CIDEr with the following settings")
print("*****************************")
print("Reference File:%s" % (refName))
print("Candidate File:%s" % (candName))
print("Result File:%s" % (resultFile))
print("IDF:%s" % (df_mode))
print("*****************************")

ref_file = os.path.join(pathToData, refName)
cand_file = os.path.join(pathToData, candName)

ref_list = json.load(open(ref_file))
cand_list = json.load(open(cand_file))

gts = {}
for ref in ref_list:
    if ref['image_id'] in gts:
        gts[ref['image_id']].append(ref['caption'])
    else:
        gts[ref['image_id']] = [ref['caption']]

# calculate cider scores
scorer = ciderEval(gts, cand_list, df_mode)
# scores: dict of list with key = metric and value = score given to each
# candidate
scores = scorer.evaluate()


# In[7]:

# scores['CIDEr'] contains CIDEr scores in a list for each candidate
# scores['CIDErD'] contains CIDEr-D scores in a list for each candidate

with open(resultFile, 'w') as outfile:
    json.dump(scores, outfile)
