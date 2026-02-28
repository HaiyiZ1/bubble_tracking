Targets to be obtained:
1) time frame;
2) per bubble area;
3) Moving velocity vector

Pipeline:
1.Preprocess picture frames (e.g., apply CLAHE to normalise contrast, Gaussian blur to denoise), check `samples/picutre_automatrix.py`.

2.Compute dense optical flow (velocity field)
From each pair $t \rightarrow t + 1$, compute displace field
Covert to velocity based on the timeframe
Output: pixel-level velocity map

3.Instance segmentation
Run Sam 2 on each frame to do segmentation
Obtain masks $M^t_i$ for each bubble instance
Assign temporary instance IDs (this is per-frame only) 
Compute bubble area from mask

4.Tracking / ID association
Match bubbles across frames using overlap or flow-guided matching (The overlap method probably won’t work due to the high moving velocity and limited frames)
Build persistent IDs over time
Detect events: merge/split

5.Per-bubble velocity computation
For bubble iat frame t: Apply mask $M^t_i$  to velocity field $t \rightarrow t + 1$
Compute median velocity inside mask

6.Area measurement
Count pixels in each mask
Convert to physical area-- Scale: 100 mm=2335 pixels

For the ID generate, for each merge and spilt we can generate a new ID. If bubble disappears, the ID terminates; do not reuse ID.
