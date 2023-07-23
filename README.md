# Deep-Encode-TUB

Requirements:

have all videos for training in one folder, have videos for prediction/evaluation in a different folder

1. run prod/create_folders.ipynb
2. run metadata extraction
3. run prod/split_scenes.ipynb
4. run prod/labeling_prod.ipynb
5. run feature extraction
6. run training and prediction
7. run evaluation


## Description CSV files:

- extract_metadata.py
  -- metadata.csv Columns: VideoName,	Resolution,	FPS,	PIX_FMT
- extract_features.py
  --features.csv  Columns:
- labeling.py (CHECK)
  -- labels.csv 
- merge_features_labes.py
  -- features_and_labels.csv Columns: Name	Bitrate	width	height	spatial_info	temporal_info	fps	entropy	Label
- dt_prediction.py
 -- predictions.csv Columns: Name	Bitrate	width	height	spatial_info	temporal_info	fps	entropy	Predicted bitrate
- DT_train_test_part.py
  -- predictions_for_eval.csv Columns: Bitrate,width,height,spatial_info,temporal_info,fps,entropy,Name,Label,Prediction
