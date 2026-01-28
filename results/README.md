# Peer-Prediction-Data

This repository contains the full experiment results under the following setting:
- Participants: Deceptive Llama 3.1 8B (x1), Honest Llama 3.1 8B (x1)
- Jury: SmolLM 135M (x1)

These results can be replicated by running the following command:
```bash
NUM_MODELS=2 NUM_DECEPTIVE_MODELS=1 PREDICTOR_MODEL_ID=6 SORT=1 python3 ./run_experiment_deceptivewitness.py
PREDICTOR_MODEL_ID=6 NUM_MODELS=2 NUM_DECEPTIVE_MODELS=1 python3 ./run_experiment_llmasajudge.py
```

Files are compressed to fit within Github's file size limit.

### `llmasajudge-...` file

The `llmasajudge-...` file contains the scores given by the jury to the responses of the participants. 

![llmasajudge](llmasajudge-screenshot.png)

The `sorted_reports` field of each dict contains the participants sorted in descending order of their scores. Each participant is represented by a 4-element list:
- The first element is the participant's ID. (0 is deceptive, 1 is honest)
- The second element is the participant's score (<=10).
- The third element is the participant's response. To avoid leaking defendant response when defendant and informant are the same model, each participant has two i.i.d. generated responses, where the first one is used when it serves as the defendant and the second one is used when it serves as the informant. The i.i.d.-ness guarantees that doing so won't bias the results.
- The fourth element is a random value in [0,1] used to break ties.

**In addition to the results with SmolLM 135M as jury, we also share results with Qwen2.5 7B as jury, which is when LLM-as-a-Judge performs at its best.**

### `logprobs-...` and `sorted_reports-...` files

The `sorted_reports-...` file contains the scores given by SmolLM 135M to the responses of the participants, in a similar format to the `llmasajudge-...` file. The `logprobs-...` file contains the log probabilities for each defendant-juror-informant triplet, as supplementary information.

![sorted_reports](sorted_reports-screenshot.png)

In `sorted_reports-...`, Each participant is represented by a 3-element list:
- The first element is the participant's ID. The one with "hyperdeceptive" in its name is the deceptive participant.
- The second element is the participant's peer prediction score.
- The third element is the participant's i.i.d. pair of responses.

### On the missing `sorted_reports` fields

In both LLM-as-a-Judge and peer prediction experiments, when the juror encounters a "context window exceeds" error, the question is skipped and no `sorted_reports` field is produced. Note though, this doesn't explain the trends we observe, as context window lengths (and the set of questions being skipped) are the same across all SmolLM jurors and across all Qwen jurors respectively.

You can feel free to suppress the "context window exceeds" error by `PYTHONWARNINGS=ignore`, as those are automatically handled by the code.