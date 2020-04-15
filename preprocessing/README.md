# Preprocessing Step

1. [OK] process_db (attraction_db_orig.json->attraction_db.json) create_delex_data.py python3
2. [OK] data.json, delex.json multiwoz_delex_data.py python2
3. [OK] create_vocab (vocab.json, act_ontology.json) create_delex_data.py python3
   [OK] create act vocab (act_vocab.json, train.json) create_act_vocab.py python3
4. [OK] train.tsv preprocess_data_for_predictor python3
