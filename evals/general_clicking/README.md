
# Evaluations

To evaluate the performance of different models on the task of clicking on UI objects we use the test-split of the TideUI dataset, which is the same as the WaveUI dataset, but converted to points. We run predictions over all examples using different models and calculate the distance between the predictions and the ground-truth, normalizing by the image diagonal to account for different image resolutions.

In a preliminary run of the first 1024 examples in the test-split we obtained the following average normalized distances:

Claude:

Average normalized distance: 0.0647838056875056

Molmo:

Average normalized distance: 0.07923065648289333